const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let tool = 'pencil';
let brushSize = 5;
let color = '#000000';
let historyStack = [];

ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.lineWidth = brushSize;
ctx.strokeStyle = color;
ctx.fillStyle = '#ffffff';
ctx.fillRect(0, 0, canvas.width, canvas.height);

function saveState() {
    historyStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    if (historyStack.length > 20) historyStack.shift();
}

saveState();

function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

canvas.addEventListener('mousedown', e => {
    isDrawing = true;
    const pos = getPos(e);
    lastX = pos.x;
    lastY = pos.y;
    if (tool === 'eraser') {
        ctx.globalCompositeOperation = 'destination-out';
    } else {
        ctx.globalCompositeOperation = 'source-over';
    }
});

canvas.addEventListener('mousemove', e => {
    if (!isDrawing) return;
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    lastX = pos.x;
    lastY = pos.y;
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
    saveState();
});

canvas.addEventListener('mouseleave', () => isDrawing = false);

document.querySelectorAll('.tool-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        tool = btn.dataset.tool;
    });
});

document.getElementById('brushSize').addEventListener('input', e => {
    brushSize = parseInt(e.target.value);
    ctx.lineWidth = brushSize;
    document.getElementById('sizeVal').textContent = brushSize;
});

document.getElementById('drawColor').addEventListener('input', e => {
    color = e.target.value;
    ctx.strokeStyle = color;
});

function clearCanvas() {
    if (!confirm('Clear the entire canvas?')) return;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    saveState();
}

function undoCanvas() {
    if (historyStack.length > 1) {
        historyStack.pop();
        const prev = historyStack[historyStack.length - 1];
        ctx.putImageData(prev, 0, 0);
    }
}

function downloadDrawing() {
    const link = document.createElement('a');
    link.download = 'scribble-drawing.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
}
