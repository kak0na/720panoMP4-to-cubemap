#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pano_to_cubefaces_gui.py

功能:
 - 选择 360° equirectangular MP4 (H.265) 视频
 - 每 N 帧抽一帧（PNG）
 - 将每帧切成 6 个 cubemap faces (right, left, top, bottom, front, back)
 - 保存每个 face（保存前统一顺时针旋转 180°）
 - GUI: 开始 / 暂停/继续 / 取消 / 打开输出文件夹
 - 抽帧临时目录优先创建在 输出目录 下（避免系统盘空间不足），失败则回退系统临时目录
 - 抽帧过程中在日志输出进度（例如：抽帧：100/2381）
 - 在检测视频总帧数前会立即输出一条日志，避免看起来卡住
"""

import os
import sys
import subprocess
import tempfile
import threading
import time
import shutil
import math
from pathlib import Path

import numpy as np
import cv2

from PyQt5 import QtWidgets, QtCore

# --------------------------
# EQUIRECTANGULAR -> CUBEMAP
# --------------------------
def equirectangular_to_cubemap_faces(img: np.ndarray):
    H, W = img.shape[:2]
    s = max(1, int(round(H / 2.0)))
    uv = np.linspace(-1, 1, s, dtype=np.float32)
    u, v = np.meshgrid(uv, -uv)

    def face_to_equi(vec_x, vec_y, vec_z):
        norm = np.sqrt(vec_x * vec_x + vec_y * vec_y + vec_z * vec_z)
        x = vec_x / norm
        y = vec_y / norm
        z = vec_z / norm
        lon = np.arctan2(z, x)
        lat = np.arcsin(y)
        uf = (lon / (2.0 * np.pi) + 0.5) * (W - 1)
        vf = (0.5 - lat / np.pi) * (H - 1)
        return uf.astype(np.float32), vf.astype(np.float32)

    faces = {}
    rx, ry, rz = 1.0, -v, -u
    ux, vx = face_to_equi(rx, ry, rz)
    faces['right'] = cv2.remap(img, ux, vx, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    lx, ly, lz = -1.0, -v, u
    ux, vx = face_to_equi(lx, ly, lz)
    faces['left'] = cv2.remap(img, ux, vx, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    tx, ty, tz = u, 1.0, v
    ux, vx = face_to_equi(tx, ty, tz)
    faces['top'] = cv2.remap(img, ux, vx, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    bx, by, bz = u, -1.0, -v
    ux, vx = face_to_equi(bx, by, bz)
    faces['bottom'] = cv2.remap(img, ux, vx, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    fx, fy, fz = u, -v, 1.0
    ux, vx = face_to_equi(fx, fy, fz)
    faces['front'] = cv2.remap(img, ux, vx, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    bx2, by2, bz2 = -u, -v, -1.0
    ux, vx = face_to_equi(bx2, by2, bz2)
    faces['back'] = cv2.remap(img, ux, vx, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    return faces

# --------------------------
# helpers
# --------------------------
def has_ffmpeg() -> bool:
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def get_total_frames(video_path: Path) -> int | None:
    """
    Try to get total frames using ffprobe (preferred). Fall back to cv2 if needed.
    Return int or None if unknown.
    """
    # try ffprobe
    try:
        if shutil.which("ffprobe"):
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames",
                "-of", "default=nokey=1:noprint_wrappers=1",
                str(video_path)
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
            out = proc.stdout.strip()
            if out and out.isdigit():
                return int(out)
    except Exception:
        pass

    # fallback to cv2
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if cnt and cnt > 0 and not math.isnan(cnt):
                return int(cnt)
    except Exception:
        pass

    return None

# --------------------------
# ConvertWorker - runs in QThread
# --------------------------
class ConvertWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)   # 0..100
    log = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal(bool, str)

    def __init__(self, video_path: str, step: int, out_dir: str, parent=None):
        super().__init__(parent)
        self.video_path = Path(video_path)
        self.step = int(step)
        self.out_dir = Path(out_dir)
        self.cancel_requested = False
        self.pause_event = threading.Event()  # set => paused
        self._ffmpeg_proc = None
        self._tmpdir_path: Path | None = None

    def request_cancel(self):
        self.cancel_requested = True
        self.log.emit("取消请求已发送。正在尝试中止...")
        proc = self._ffmpeg_proc
        if proc:
            try:
                proc.terminate()
                self.log.emit("已向 ffmpeg 发送 terminate 请求。")
            except Exception as e:
                self.log.emit(f"尝试终止 ffmpeg 时出错: {e}")

    def toggle_pause(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.log.emit("继续处理中...")
        else:
            self.pause_event.set()
            self.log.emit("已暂停。")

    def _wait_while_paused_or_cancel(self):
        while self.pause_event.is_set():
            if self.cancel_requested:
                raise RuntimeError("已取消")
            time.sleep(0.15)
        if self.cancel_requested:
            raise RuntimeError("已取消")

    def _make_temp_dir(self) -> Path:
        """
        Try create temp dir under output dir (same disk) to avoid system C: space issue;
        fallback to current working dir or system temp.
        """
        try:
            if self.out_dir:
                self.out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.log.emit(f"警告：无法创建输出目录 {self.out_dir}，将回退到当前工作目录或系统临时目录。错误: {e}")

        try:
            base = self.out_dir if self.out_dir and self.out_dir.exists() else Path.cwd()
            tmp = base / f".pano_tmp_frames_{int(time.time())}"
            tmp.mkdir(parents=False, exist_ok=False)
            return tmp
        except Exception as e:
            self.log.emit(f"无法在首选位置创建临时目录（{e}），尝试系统临时目录。")
            tmp = Path(tempfile.mkdtemp(prefix="pano_tmp_"))
            return tmp

    def _cleanup_tmpdir(self):
        try:
            if self._tmpdir_path and self._tmpdir_path.exists():
                shutil.rmtree(self._tmpdir_path, ignore_errors=True)
                self.log.emit(f"已清理临时目录：{self._tmpdir_path}")
        except Exception as e:
            self.log.emit(f"清理临时目录时出错: {e}")

    def run(self):
        try:
            self.log.emit(f"开始转换：\n  视频={self.video_path}\n  step={self.step}\n  输出={self.out_dir}")
            # prepare tmp dir (prefer output dir)
            self._tmpdir_path = self._make_temp_dir()
            frames_tmp = Path(self._tmpdir_path)
            self.log.emit(f"临时帧目录：{frames_tmp}")

            # --- NEW: immediately log that we're detecting total frames so UI isn't idle-looking ---
            self.log.emit("正在检测视频总帧数...（这可能需要几秒钟）")
            # try to compute total frames and expected extracted count
            total_input_frames = get_total_frames(self.video_path)
            if total_input_frames is not None:
                expected_extracted = math.ceil(total_input_frames / max(1, self.step))
                self.log.emit(f"检测到视频总帧数: {total_input_frames}，预计抽出帧数: {expected_extracted}")
            else:
                expected_extracted = None
                self.log.emit("无法确定视频总帧数，抽帧时只显示已抽取数量。")

            # extraction: prefer ffmpeg
            self.log.emit("开始抽帧...")
            if has_ffmpeg():
                self.log.emit("使用 ffmpeg 抽帧（优先，支持 h265）")
                expr = f"select='not(mod(n\\,{self.step}))'"
                out_pattern = str(frames_tmp / 'frame_%06d.png')
                cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(self.video_path),
                       '-vf', expr, '-vsync', 'vfr', out_pattern]
                try:
                    # start ffmpeg and capture stderr for diagnostics
                    self._ffmpeg_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    last_count = -1
                    last_report_time = 0.0
                    poll_interval = 0.5  # seconds, how often to check file count
                    while True:
                        # check cancel/pause
                        if self.cancel_requested:
                            try:
                                self._ffmpeg_proc.terminate()
                            except Exception:
                                pass
                            try:
                                self._ffmpeg_proc.wait(timeout=5)
                            except Exception:
                                pass
                            raise RuntimeError("已取消")

                        if self.pause_event.is_set():
                            time.sleep(0.2)
                            continue

                        # count files for progress
                        now = time.time()
                        if now - last_report_time >= poll_interval:
                            try:
                                cnt = len(list(frames_tmp.glob('frame_*.png')))
                            except Exception:
                                cnt = 0
                            if cnt != last_count:
                                last_count = cnt
                                if expected_extracted:
                                    self.log.emit(f"抽帧: {cnt}/{expected_extracted}")
                                else:
                                    self.log.emit(f"抽帧: {cnt}")
                            last_report_time = now

                        ret = self._ffmpeg_proc.poll()
                        if ret is not None:
                            # process ended; read stderr
                            try:
                                _, stderr = self._ffmpeg_proc.communicate(timeout=1)
                                stderr_text = stderr.decode(errors='ignore') if stderr else ""
                            except Exception:
                                stderr_text = ""
                            if ret != 0:
                                snippet = stderr_text.strip()[:4000]
                                self.log.emit(f"ffmpeg 非零退出码 {ret}。ffmpeg stderr（节选）:\n{snippet}")
                                if "No space left on device" in stderr_text or "No such file or directory" in stderr_text:
                                    raise RuntimeError(f"ffmpeg 返回非零退出码: {ret}（可能磁盘空间不足或路径问题）。ffmpeg 输出: {snippet}")
                                else:
                                    raise RuntimeError(f"ffmpeg 返回非零退出码: {ret}。ffmpeg 输出（节选）:\n{snippet}")
                            # final count log
                            try:
                                final_cnt = len(list(frames_tmp.glob('frame_*.png')))
                            except Exception:
                                final_cnt = 0
                            if expected_extracted:
                                self.log.emit(f"抽帧完成：{final_cnt}/{expected_extracted}")
                            else:
                                self.log.emit(f"抽帧完成：{final_cnt}")
                            break

                        time.sleep(0.1)
                finally:
                    self._ffmpeg_proc = None
            else:
                # fallback to OpenCV extraction with in-loop progress
                self.log.emit("未检测到 ffmpeg，使用 OpenCV 回退抽帧（可能无法解码 h265）")
                cap = cv2.VideoCapture(str(self.video_path))
                if not cap.isOpened():
                    raise RuntimeError("无法打开视频：OpenCV VideoCapture 失败（没有 ffmpeg 或编解码器不支持 h265）")
                idx = 0
                out_idx = 1
                last_report = 0
                while True:
                    self._wait_while_paused_or_cancel()
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if idx % self.step == 0:
                        fname = frames_tmp / f"frame_{out_idx:06d}.png"
                        cv2.imwrite(str(fname), frame)
                        out_idx += 1
                        # report progress every 10 frames to avoid spam
                        if out_idx - 1 - last_report >= 10:
                            try:
                                cnt = len(list(frames_tmp.glob('frame_*.png')))
                            except Exception:
                                cnt = out_idx - 1
                            if expected_extracted:
                                self.log.emit(f"抽帧: {cnt}/{expected_extracted}")
                            else:
                                self.log.emit(f"抽帧: {cnt}")
                            last_report = out_idx - 1
                    idx += 1
                cap.release()
                final_cnt = len(list(frames_tmp.glob('frame_*.png')))
                if expected_extracted:
                    self.log.emit(f"抽帧完成：{final_cnt}/{expected_extracted}")
                else:
                    self.log.emit(f"抽帧完成：{final_cnt}")

            # list extracted frames
            frames = sorted([p for p in frames_tmp.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
            if len(frames) == 0:
                raise RuntimeError("未能抽取到任何帧。请检查视频、抽帧间隔（step）以及临时目录磁盘空间。")
            total = len(frames)
            self.log.emit(f"抽帧完成：共 {total} 帧。开始切分并保存 6 面...")

            # ensure output folder exists
            try:
                self.out_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.log.emit(f"无法创建输出目录 {self.out_dir}：{e}")

            for i, fpath in enumerate(frames, start=1):
                self._wait_while_paused_or_cancel()
                if self.cancel_requested:
                    raise RuntimeError("已取消")

                img = cv2.imread(str(fpath))
                if img is None:
                    self.log.emit(f"警告：无法读取帧 {fpath.name}，跳过。")
                    continue

                faces = equirectangular_to_cubemap_faces(img)
                base = f"frame_{i:06d}"
                for key, face_img in faces.items():
                    # rotate 180 degrees clockwise before saving
                    rotated = cv2.rotate(face_img, cv2.ROTATE_180)
                    out_name = self.out_dir / f"{base}_{key}.png"
                    try:
                        cv2.imwrite(str(out_name), rotated)
                    except Exception as e:
                        self.log.emit(f"写出文件失败: {out_name}，错误: {e}")
                        raise

                self.progress.emit(int(i * 100 / total))
                self.log.emit(f"处理 {i}/{total}：已保存 6 个面 ({fpath.name})")

            self.progress.emit(100)
            self.finished_signal.emit(True, "完成！所有帧已处理并保存。")
        except Exception as e:
            # emit error
            self.log.emit(f"错误：{e}")
            self.finished_signal.emit(False, str(e))
        finally:
            # always cleanup tmpdir if created
            try:
                self._cleanup_tmpdir()
            except Exception:
                pass

# --------------------------
# PyQt5 GUI
# --------------------------
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("360全景视频→切6图 工具")
        self.resize(780, 460)
        self.worker: ConvertWorker | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # video file
        file_layout = QtWidgets.QHBoxLayout()
        self.input_edit = QtWidgets.QLineEdit()
        self.btn_browse = QtWidgets.QPushButton("选择视频文件")
        self.btn_browse.clicked.connect(self.browse_file)
        file_layout.addWidget(QtWidgets.QLabel("视频文件:"))
        file_layout.addWidget(self.input_edit)
        file_layout.addWidget(self.btn_browse)

        # step
        step_layout = QtWidgets.QHBoxLayout()
        self.step_spin = QtWidgets.QSpinBox()
        self.step_spin.setMinimum(1)
        self.step_spin.setValue(30)
        self.step_spin.setMaximum(10000000)
        step_layout.addWidget(QtWidgets.QLabel("每多少帧抽一帧 (N)："))
        step_layout.addWidget(self.step_spin)
        step_layout.addStretch()

        # output folder + open button
        out_layout = QtWidgets.QHBoxLayout()
        self.out_edit = QtWidgets.QLineEdit()
        self.btn_out = QtWidgets.QPushButton("选择输出文件夹")
        self.btn_out.clicked.connect(self.browse_out)
        self.btn_open_out = QtWidgets.QPushButton("打开输出文件夹")
        self.btn_open_out.clicked.connect(self.open_output_folder)
        out_layout.addWidget(QtWidgets.QLabel("输出文件夹:"))
        out_layout.addWidget(self.out_edit)
        out_layout.addWidget(self.btn_out)
        out_layout.addWidget(self.btn_open_out)

        # control buttons
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("开始")
        self.btn_start.clicked.connect(self.start)
        self.btn_pause = QtWidgets.QPushButton("暂停")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.on_pause_clicked)
        self.btn_cancel = QtWidgets.QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.on_cancel_clicked)
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_pause)
        ctrl_layout.addWidget(self.btn_cancel)
        ctrl_layout.addStretch()

        # progress + log
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)

        layout.addLayout(file_layout)
        layout.addLayout(step_layout)
        layout.addLayout(out_layout)
        layout.addLayout(ctrl_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QtWidgets.QLabel("日志:"))
        layout.addWidget(self.log_text)

    def browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.mov *.mkv *.avi);;All Files (*)")
        if path:
            self.input_edit.setText(path)

    def browse_out(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出文件夹", "")
        if path:
            self.out_edit.setText(path)

    def open_output_folder(self):
        folder = self.out_edit.text().strip()
        if not folder:
            QtWidgets.QMessageBox.warning(self, "错误", "请先选择或填写输出文件夹路径。")
            return
        path = Path(folder)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "错误", f"无法创建输出文件夹: {e}")
                return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(path)])
            else:
                subprocess.run(["xdg-open", str(path)])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "打开失败", f"无法打开文件夹：{e}")

    def append_log(self, text: str):
        self.log_text.appendPlainText(text)
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)

    def set_running_state(self, running: bool):
        self.input_edit.setEnabled(not running)
        self.btn_browse.setEnabled(not running)
        self.step_spin.setEnabled(not running)
        self.out_edit.setEnabled(not running)
        self.btn_out.setEnabled(not running)
        # keep open output always enabled
        self.btn_start.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        self.btn_cancel.setEnabled(running)
        if not running:
            self.btn_pause.setText("暂停")

    def start(self):
        video = self.input_edit.text().strip()
        out = self.out_edit.text().strip()
        step = int(self.step_spin.value())
        if not video or not Path(video).is_file():
            QtWidgets.QMessageBox.warning(self, "错误", "请选择有效的视频文件。")
            return
        if not out:
            QtWidgets.QMessageBox.warning(self, "错误", "请选择输出文件夹。")
            return

        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.set_running_state(True)

        self.worker = ConvertWorker(video, step, out)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.append_log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()
        self.append_log("任务已开始。可使用 暂停/继续 或 取消。 打开输出文件夹在运行时仍可点击。")

    def on_pause_clicked(self):
        if not self.worker:
            return
        self.worker.toggle_pause()
        if self.worker.pause_event.is_set():
            self.btn_pause.setText("继续")
        else:
            self.btn_pause.setText("暂停")

    def on_cancel_clicked(self):
        if not self.worker:
            return
        self.worker.request_cancel()
        self.btn_cancel.setEnabled(False)
        self.append_log("已发送取消请求，正在尝试中止...")

    @QtCore.pyqtSlot(bool, str)
    def on_finished(self, ok: bool, msg: str):
        self.set_running_state(False)
        if ok:
            QtWidgets.QMessageBox.information(self, "完成", msg)
            self.append_log("完成: " + msg)
        else:
            QtWidgets.QMessageBox.critical(self, "失败", msg)
            self.append_log("失败: " + msg)
        self.worker = None

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
