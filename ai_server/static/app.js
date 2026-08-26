const $ = (selector) => document.querySelector(selector);

const uploadForm = $("#uploadForm");
const videoFile = $("#videoFile");
const uploadButton = $("#uploadButton");
const statusEl = $("#status");
const sourceVideo = $("#sourceVideo");
const roiCanvas = $("#roiCanvas");
const emptyState = $("#emptyState");
const videoMeta = $("#videoMeta");
const initFrameInput = $("#initFrame");
const currentTimeInput = $("#currentTime");
const pointXInput = $("#pointX");
const pointYInput = $("#pointY");
const bboxWInput = $("#bboxW");
const bboxHInput = $("#bboxH");
const maxSecondsInput = $("#maxSeconds");
const stabilizeInput = $("#stabilize");
const selectButton = $("#selectButton");
const clearButton = $("#clearButton");
const trackButton = $("#trackButton");
const selectionHint = $("#selectionHint");
const onlineUpdateInput = $("#onlineUpdate");
const onlineUpdateValue = $("#onlineUpdateValue");
const kltAcceptConfInput = $("#kltAcceptConf");
const kltAcceptConfValue = $("#kltAcceptConfValue");
const recoveryConfInput = $("#recoveryConf");
const recoveryConfValue = $("#recoveryConfValue");
const updateConfInput = $("#updateConf");
const updateConfValue = $("#updateConfValue");
const updateConfField = $("#updateConfField");
const searchRadiusInput = $("#searchRadius");
const searchRadiusValue = $("#searchRadiusValue");
const resetTuning = $("#resetTuning");
const resultSection = $("#resultSection");
const resultSummary = $("#resultSummary");
const resultMetrics = $("#resultMetrics");
const overlayArea = $("#overlayArea");
const overlayVideo = $("#overlayVideo");
const debugPanel = $("#debugPanel");
const downloadTrack = $("#downloadTrack");
const downloadTrajectory = $("#downloadTrajectory");
const downloadMetrics = $("#downloadMetrics");
const downloadOverlay = $("#downloadOverlay");

let preparedVideo = null;
let selectedBox = null;
let selectionFrame = null;
let selectionMode = false;
let dragStart = null;
let dragCurrent = null;
let hoverPoint = null;

const tuningDefaults = {
  kltAcceptConf: 0.4,
  recoveryConf: 0.58,
  updateConf: 0.76,
  searchRadius: 2.5,
  onlineUpdate: false,
};

uploadForm.addEventListener("submit", uploadVideo);
selectButton.addEventListener("click", beginSelection);
clearButton.addEventListener("click", () => {
  clearSelection();
  beginSelection();
});
trackButton.addEventListener("click", runTracking);
resetTuning.addEventListener("click", resetTuningValues);

[
  kltAcceptConfInput,
  recoveryConfInput,
  updateConfInput,
  searchRadiusInput,
].forEach((input) => input.addEventListener("input", syncTuning));
onlineUpdateInput.addEventListener("change", syncTuning);

sourceVideo.addEventListener("loadedmetadata", () => {
  emptyState.classList.add("hidden");
  syncCanvasSize();
  syncTimeFromVideo();
  drawSelection();
});
sourceVideo.addEventListener("timeupdate", syncTimeFromVideo);
sourceVideo.addEventListener("seeked", () => {
  syncTimeFromVideo();
  drawSelection();
});
sourceVideo.addEventListener("play", () => {
  if (selectionMode) sourceVideo.pause();
});

window.addEventListener("resize", drawSelection);

currentTimeInput.addEventListener("change", () => {
  if (!preparedVideo) return;
  const requested = Number(currentTimeInput.value || 0);
  if (Number.isFinite(requested)) {
    sourceVideo.currentTime = clamp(requested, 0, sourceVideo.duration || requested);
  }
});

initFrameInput.addEventListener("change", () => {
  if (!preparedVideo) return;
  const frame = Math.max(0, Math.round(Number(initFrameInput.value || 0)));
  sourceVideo.currentTime = frame / Math.max(Number(preparedVideo.metadata.fps), 1e-6);
  if (selectedBox) selectionFrame = frame;
});

[pointXInput, pointYInput, bboxWInput, bboxHInput].forEach((input) => {
  input.addEventListener("change", updateBoxFromInputs);
});

roiCanvas.addEventListener("pointerdown", (event) => {
  if (!selectionMode || !preparedVideo) return;
  event.preventDefault();
  event.stopPropagation();
  sourceVideo.pause();
  selectionFrame = currentFrameIndex();
  dragStart = canvasEventToVideoPoint(event);
  dragCurrent = dragStart;
  hoverPoint = dragStart;
  roiCanvas.setPointerCapture(event.pointerId);
  drawSelection();
});

roiCanvas.addEventListener("pointermove", (event) => {
  if (!selectionMode || !preparedVideo) return;
  event.preventDefault();
  hoverPoint = canvasEventToVideoPoint(event);
  if (dragStart) dragCurrent = hoverPoint;
  drawSelection();
});

roiCanvas.addEventListener("pointerup", (event) => {
  if (!selectionMode || !dragStart) return;
  event.preventDefault();
  event.stopPropagation();
  dragCurrent = canvasEventToVideoPoint(event);
  finishSelection();
  if (roiCanvas.hasPointerCapture(event.pointerId)) {
    roiCanvas.releasePointerCapture(event.pointerId);
  }
});

roiCanvas.addEventListener("pointercancel", () => {
  dragStart = null;
  dragCurrent = null;
  drawSelection();
});

resetTuningValues();
resetUi();

async function uploadVideo(event) {
  event.preventDefault();
  const file = videoFile.files[0];
  if (!file) {
    setStatus("영상을 선택하세요", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  resetUi({ keepFile: true });
  setStatus("업로드 중", "busy");
  uploadButton.disabled = true;

  try {
    const response = await fetch("/api/videos/upload", {
      method: "POST",
      body: formData,
    });
    const data = await readResponse(response);
    preparedVideo = data;
    sourceVideo.src = data.download_urls.source_video;
    sourceVideo.load();
    renderVideoMeta(data.metadata);
    enableControls(true);
    setStatus("영상 준비 완료");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    uploadButton.disabled = false;
  }
}

function beginSelection() {
  if (!preparedVideo) return;
  sourceVideo.pause();
  selectionMode = true;
  selectionFrame = currentFrameIndex();
  dragStart = null;
  dragCurrent = null;
  hoverPoint = null;
  initFrameInput.value = String(selectionFrame);
  currentTimeInput.value = sourceVideo.currentTime.toFixed(2);
  roiCanvas.classList.add("selecting");
  selectionHint.classList.add("active");
  selectionHint.textContent =
    "프레임이 고정되었습니다. 객체 주변을 드래그하세요. 짧게 클릭하면 32px 박스를 만듭니다.";
  selectButton.disabled = true;
  trackButton.disabled = true;
  setStatus(`객체 선택 중 · frame ${selectionFrame}`, "busy");
  drawSelection();
}

function finishSelection() {
  const raw = boxFromPoints(dragStart, dragCurrent);
  const box = raw.w < 4 || raw.h < 4
    ? { x: dragStart.x - 16, y: dragStart.y - 16, w: 32, h: 32 }
    : raw;
  selectedBox = clampBox(box, sourceVideo.videoWidth, sourceVideo.videoHeight);
  selectionFrame = selectionFrame ?? currentFrameIndex();
  syncInputsFromBox();

  selectionMode = false;
  dragStart = null;
  dragCurrent = null;
  hoverPoint = null;
  roiCanvas.classList.remove("selecting");
  selectionHint.classList.remove("active");
  selectionHint.textContent =
    `frame ${selectionFrame} · ROI ${selectedBox.w.toFixed(0)}×${selectedBox.h.toFixed(0)} 고정`;
  selectButton.disabled = false;
  clearButton.disabled = false;
  trackButton.disabled = false;
  setStatus("객체 선택 완료");
  drawSelection();
}

function clearSelection() {
  selectedBox = null;
  selectionFrame = null;
  selectionMode = false;
  dragStart = null;
  dragCurrent = null;
  hoverPoint = null;
  pointXInput.value = "";
  pointYInput.value = "";
  bboxWInput.value = "32";
  bboxHInput.value = "32";
  roiCanvas.classList.remove("selecting");
  selectionHint.classList.remove("active");
  selectionHint.textContent =
    "영상을 멈춘 뒤 객체 선택을 누르고 비행체 주변을 드래그하세요.";
  enableControls(Boolean(preparedVideo));
  drawSelection();
}

async function runTracking() {
  if (!preparedVideo || !selectedBox) return;
  sourceVideo.pause();
  const tuning = readTuning();
  const duration = Number(maxSecondsInput.value || 0);
  const payload = {
    source_video_id: preparedVideo.source_video_id,
    video_path: preparedVideo.video_path,
    init_frame_index: Number(selectionFrame ?? currentFrameIndex()),
    target_bbox: [selectedBox.x, selectedBox.y, selectedBox.w, selectedBox.h],
    stabilize: stabilizeInput.checked,
    resize_width: 1280,
    write_overlay: true,
    tuning: {
      klt_accept_conf: tuning.kltAcceptConf,
      recovery_conf: tuning.recoveryConf,
      update_conf: tuning.updateConf,
      search_radius_multiplier: tuning.searchRadius,
      online_update_enabled: tuning.onlineUpdate,
    },
  };
  if (Number.isFinite(duration) && duration > 0) payload.max_seconds = duration;

  setStatus("추적 계산 중", "busy");
  trackButton.disabled = true;
  clearResult();
  try {
    const response = await fetch("/api/tracks/manual", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await readResponse(response);
    renderResult(data);
    setStatus("추적 완료");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    trackButton.disabled = false;
  }
}

function renderResult(data) {
  const track = data.tracks?.[0];
  const quality = track?.quality || {};
  const metrics = data.metrics || {};
  const urls = data.download_urls || {};

  resultSection.classList.remove("hidden");
  resultSummary.textContent = [
    `${formatInt(quality.num_points)} observed`,
    `${formatPercent(metrics.visible_ratio)} visible`,
    quality.track_stability || "unknown",
    metrics.method || "manual_roi",
  ].join(" · ");

  renderMetrics([
    ["observed", quality.num_points],
    ["mean conf", formatNumber(quality.mean_conf)],
    ["missing", formatPercent(quality.missing_ratio)],
    ["KLT", metrics.tracking_source_counts?.klt],
    ["appearance", metrics.tracking_source_counts?.appearance],
    ["motion", metrics.tracking_source_counts?.motion],
  ]);

  setLink(downloadTrack, urls.track_sequence);
  setLink(downloadTrajectory, urls.trajectory);
  setLink(downloadMetrics, urls.metrics);
  setLink(downloadOverlay, urls.overlay);

  overlayArea.classList.toggle("hidden", !urls.overlay);
  if (urls.overlay) {
    overlayVideo.src = urls.overlay;
    overlayVideo.load();
  }
  debugPanel.textContent = JSON.stringify(
    {
      run_id: data.metadata?.run_id,
      init_frame_index: data.metadata?.init_frame_index,
      target_bbox: selectedBox,
      coordinate_mode: data.metadata?.coordinate_mode,
      quality,
      metrics,
      first_point: track?.history?.[0],
      last_point: track?.history?.at(-1),
    },
    null,
    2,
  );
  resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderMetrics(items) {
  resultMetrics.replaceChildren(
    ...items.map(([label, value]) => {
      const item = document.createElement("div");
      const term = document.createElement("dt");
      const detail = document.createElement("dd");
      term.textContent = label;
      detail.textContent = value ?? "-";
      item.append(term, detail);
      return item;
    }),
  );
}

function drawSelection() {
  syncCanvasSize();
  const context = roiCanvas.getContext("2d");
  context.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
  if (!sourceVideo.videoWidth || !sourceVideo.videoHeight) return;

  const scaleX = roiCanvas.width / sourceVideo.videoWidth;
  const scaleY = roiCanvas.height / sourceVideo.videoHeight;
  const preview = dragStart && dragCurrent
    ? boxFromPoints(dragStart, dragCurrent)
    : selectedBox;

  if (preview) {
    const x = preview.x * scaleX;
    const y = preview.y * scaleY;
    const width = preview.w * scaleX;
    const height = preview.h * scaleY;
    context.save();
    context.fillStyle = "rgba(243, 182, 66, 0.14)";
    context.strokeStyle = "#f3b642";
    context.lineWidth = Math.max(2, 2 * window.devicePixelRatio);
    context.fillRect(x, y, width, height);
    context.strokeRect(x, y, width, height);
    drawCrosshair(context, x + width / 2, y + height / 2);
    context.restore();
  }

  if (selectionMode && hoverPoint) {
    drawMagnifier(context, hoverPoint, scaleX, scaleY);
  }
}

function drawCrosshair(context, x, y) {
  const arm = 9 * window.devicePixelRatio;
  context.strokeStyle = "#e05a37";
  context.beginPath();
  context.moveTo(x - arm, y);
  context.lineTo(x + arm, y);
  context.moveTo(x, y - arm);
  context.lineTo(x, y + arm);
  context.stroke();
}

function drawMagnifier(context, point, scaleX, scaleY) {
  const dpr = window.devicePixelRatio;
  const size = 126 * dpr;
  const sourceSize = 38;
  const canvasX = point.x * scaleX;
  const canvasY = point.y * scaleY;
  let x = canvasX + 18 * dpr;
  let y = canvasY - size / 2;
  if (x + size > roiCanvas.width) x = canvasX - size - 18 * dpr;
  y = clamp(y, 8 * dpr, Math.max(8 * dpr, roiCanvas.height - size - 8 * dpr));
  const sourceX = clamp(point.x - sourceSize / 2, 0, sourceVideo.videoWidth - sourceSize);
  const sourceY = clamp(point.y - sourceSize / 2, 0, sourceVideo.videoHeight - sourceSize);

  context.save();
  context.imageSmoothingEnabled = false;
  context.fillStyle = "#10251f";
  context.fillRect(x - 3 * dpr, y - 3 * dpr, size + 6 * dpr, size + 6 * dpr);
  context.drawImage(sourceVideo, sourceX, sourceY, sourceSize, sourceSize, x, y, size, size);
  context.strokeStyle = "#f3b642";
  context.lineWidth = 2 * dpr;
  context.strokeRect(x, y, size, size);
  drawCrosshair(context, x + size / 2, y + size / 2);
  context.restore();
}

function updateBoxFromInputs() {
  if (!preparedVideo || !sourceVideo.videoWidth) return;
  const centerX = Number(pointXInput.value);
  const centerY = Number(pointYInput.value);
  const width = Math.max(4, Number(bboxWInput.value || 32));
  const height = Math.max(4, Number(bboxHInput.value || 32));
  if (!Number.isFinite(centerX) || !Number.isFinite(centerY)) return;

  selectedBox = clampBox(
    { x: centerX - width / 2, y: centerY - height / 2, w: width, h: height },
    sourceVideo.videoWidth,
    sourceVideo.videoHeight,
  );
  selectionFrame = Math.max(0, Math.round(Number(initFrameInput.value || 0)));
  syncInputsFromBox();
  clearButton.disabled = false;
  trackButton.disabled = false;
  drawSelection();
}

function syncInputsFromBox() {
  if (!selectedBox) return;
  pointXInput.value = (selectedBox.x + selectedBox.w / 2).toFixed(1);
  pointYInput.value = (selectedBox.y + selectedBox.h / 2).toFixed(1);
  bboxWInput.value = selectedBox.w.toFixed(1);
  bboxHInput.value = selectedBox.h.toFixed(1);
  initFrameInput.value = String(selectionFrame ?? currentFrameIndex());
}

function syncCanvasSize() {
  const rect = sourceVideo.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  roiCanvas.width = Math.max(1, Math.round(rect.width * dpr));
  roiCanvas.height = Math.max(1, Math.round(rect.height * dpr));
  roiCanvas.style.width = `${rect.width}px`;
  roiCanvas.style.height = `${rect.height}px`;
}

function canvasEventToVideoPoint(event) {
  const rect = roiCanvas.getBoundingClientRect();
  return {
    x: clamp(
      ((event.clientX - rect.left) / Math.max(rect.width, 1)) * sourceVideo.videoWidth,
      0,
      Math.max(0, sourceVideo.videoWidth - 1),
    ),
    y: clamp(
      ((event.clientY - rect.top) / Math.max(rect.height, 1)) * sourceVideo.videoHeight,
      0,
      Math.max(0, sourceVideo.videoHeight - 1),
    ),
  };
}

function boxFromPoints(start, end) {
  return {
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    w: Math.abs(end.x - start.x),
    h: Math.abs(end.y - start.y),
  };
}

function clampBox(box, frameWidth, frameHeight) {
  const width = clamp(Number(box.w), 4, Math.max(4, frameWidth));
  const height = clamp(Number(box.h), 4, Math.max(4, frameHeight));
  return {
    x: clamp(Number(box.x), 0, Math.max(0, frameWidth - width)),
    y: clamp(Number(box.y), 0, Math.max(0, frameHeight - height)),
    w: width,
    h: height,
  };
}

function syncTimeFromVideo() {
  if (!preparedVideo) return;
  currentTimeInput.value = sourceVideo.currentTime.toFixed(2);
  if (selectionFrame === null) initFrameInput.value = String(currentFrameIndex());
}

function currentFrameIndex() {
  const fps = Number(preparedVideo?.metadata?.fps || 30);
  return Math.max(0, Math.round(sourceVideo.currentTime * fps));
}

function readTuning() {
  return {
    kltAcceptConf: Number(kltAcceptConfInput.value),
    recoveryConf: Number(recoveryConfInput.value),
    updateConf: Number(updateConfInput.value),
    searchRadius: Number(searchRadiusInput.value),
    onlineUpdate: onlineUpdateInput.checked,
  };
}

function syncTuning() {
  const tuning = readTuning();
  kltAcceptConfValue.textContent = tuning.kltAcceptConf.toFixed(2);
  recoveryConfValue.textContent = tuning.recoveryConf.toFixed(2);
  updateConfValue.textContent = tuning.updateConf.toFixed(2);
  searchRadiusValue.textContent = `${tuning.searchRadius.toFixed(1)}x`;
  onlineUpdateValue.textContent = tuning.onlineUpdate ? "ON" : "OFF";
  updateConfInput.disabled = !tuning.onlineUpdate;
  updateConfField.classList.toggle("disabled", !tuning.onlineUpdate);
}

function resetTuningValues(event) {
  event?.preventDefault();
  event?.stopPropagation();
  kltAcceptConfInput.value = String(tuningDefaults.kltAcceptConf);
  recoveryConfInput.value = String(tuningDefaults.recoveryConf);
  updateConfInput.value = String(tuningDefaults.updateConf);
  searchRadiusInput.value = String(tuningDefaults.searchRadius);
  onlineUpdateInput.checked = tuningDefaults.onlineUpdate;
  syncTuning();
}

function enableControls(enabled) {
  selectButton.disabled = !enabled;
  clearButton.disabled = !(enabled && selectedBox);
  trackButton.disabled = !(enabled && selectedBox);
}

function resetUi(options = {}) {
  preparedVideo = null;
  selectedBox = null;
  selectionFrame = null;
  selectionMode = false;
  dragStart = null;
  dragCurrent = null;
  hoverPoint = null;
  sourceVideo.removeAttribute("src");
  sourceVideo.load();
  emptyState.classList.remove("hidden");
  videoMeta.textContent = "-";
  pointXInput.value = "";
  pointYInput.value = "";
  bboxWInput.value = "32";
  bboxHInput.value = "32";
  initFrameInput.value = "0";
  currentTimeInput.value = "0";
  roiCanvas.classList.remove("selecting");
  selectionHint.classList.remove("active");
  selectionHint.textContent =
    "영상을 멈춘 뒤 객체 선택을 누르고 비행체 주변을 드래그하세요.";
  enableControls(false);
  clearResult();
  if (!options.keepFile) videoFile.value = "";
  drawSelection();
}

function clearResult() {
  resultSection.classList.add("hidden");
  overlayArea.classList.add("hidden");
  overlayVideo.removeAttribute("src");
  debugPanel.textContent = "";
  resultSummary.textContent = "";
  resultMetrics.replaceChildren();
  [downloadTrack, downloadTrajectory, downloadMetrics, downloadOverlay].forEach(
    (link) => setLink(link, null),
  );
}

function renderVideoMeta(metadata) {
  videoMeta.textContent =
    `${metadata.width}×${metadata.height} · ` +
    `${Number(metadata.fps).toFixed(1)}fps · ${metadata.frame_count}f`;
}

function setLink(link, href) {
  link.href = href || "#";
  link.classList.toggle("disabled", !href);
}

function setStatus(message, state = "ready") {
  statusEl.textContent = message;
  statusEl.classList.toggle("busy", state === "busy");
  statusEl.classList.toggle("error", state === "error");
}

async function readResponse(response) {
  const data = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg).join(", ")
      : data.detail;
    throw new Error(detail || "요청을 처리하지 못했습니다.");
  }
  return data;
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "-";
}

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "-";
}

function formatInt(value) {
  const number = Number(value);
  return Number.isFinite(number) ? String(Math.round(number)) : "-";
}

function clamp(value, minimum, maximum) {
  return Math.min(Math.max(value, minimum), maximum);
}
