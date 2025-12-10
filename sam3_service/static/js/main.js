/**
 * SAM3 隐私过滤前端逻辑
 */

const API_BASE = '';  // 同源，无需前缀

// DOM 元素
const imageInput = document.getElementById('image-input');
const textPromptInput = document.getElementById('text-prompt') || { value: '' };
const previewMode = document.getElementById('preview-mode');
const blurType = document.getElementById('blur-type');
const blurStrength = document.getElementById('blur-strength');
const strengthValue = document.getElementById('strength-value');
const previewBtn = document.getElementById('preview-btn');
const applyBtn = document.getElementById('apply-btn');
const statusDiv = document.getElementById('status');

const originalPlaceholder = document.getElementById('original-placeholder');
const originalPreview = document.getElementById('original-preview');
const resultPlaceholder = document.getElementById('result-placeholder');
const resultPreview = document.getElementById('result-preview');
const regionsInfo = document.getElementById('regions-info');

let selectedFile = null;
let lastPreviewRegions = 0;

// 更新强度显示
blurStrength.addEventListener('input', () => {
    strengthValue.textContent = blurStrength.value;
});

// 图片选择
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) {
        selectedFile = null;
        previewBtn.disabled = true;
        applyBtn.disabled = true;
        originalPreview.style.display = 'none';
        originalPlaceholder.style.display = 'flex';
        return;
    }
    
    selectedFile = file;
    previewBtn.disabled = false;
    applyBtn.disabled = false;  // 允许直接应用模糊（mock 模式）
    
    // 预览原图
    const reader = new FileReader();
    reader.onload = (ev) => {
        originalPreview.src = ev.target.result;
        originalPreview.style.display = 'block';
        originalPlaceholder.style.display = 'none';
    };
    reader.readAsDataURL(file);
    
    // 清空结果
    resultPreview.style.display = 'none';
    resultPlaceholder.style.display = 'flex';
    regionsInfo.textContent = '';
    lastPreviewRegions = 0;
});

// 预览分割
previewBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // 显示加载状态
    setStatus('loading', '正在根据文本提示词进行分割预览...');
    previewBtn.disabled = true;
    applyBtn.disabled = true;
    
    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('text_prompt', textPromptInput.value || 'all objects');
        formData.append('preview_mode', previewMode ? previewMode.value : 'heatmap');
        
        const response = await fetch(`${API_BASE}/v1/segment/text_preview`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`预览请求失败: ${response.status} - ${errText}`);
        }
        
        const data = await response.json();
        
        // 显示预览结果（叠加 bbox）
        resultPreview.src = data.preview_image_base64;
        resultPreview.style.display = 'block';
        resultPlaceholder.style.display = 'none';
        
        // 显示区域信息
        const regionCount = data.applied_regions.length;
        lastPreviewRegions = regionCount;
        regionsInfo.textContent = `预览选中 ${regionCount} 个区域`;
        
        setStatus('success', '分割预览完成，请确认后点击“应用模糊”。');
        
        // 有区域时才允许应用模糊
        applyBtn.disabled = regionCount === 0;
    } catch (err) {
        console.error(err);
        // 网络错误时提示用户可以直接用"应用模糊"
        if (err.message.includes('fetch') || err.message.includes('Failed')) {
            setStatus('error', '无法连接服务器，可直接点击"应用模糊"体验 Mock 效果（中心圆形区域）');
        } else {
            setStatus('error', `错误: ${err.message}`);
        }
    } finally {
        previewBtn.disabled = !selectedFile;
        applyBtn.disabled = !selectedFile;  // 保持可用
    }
});

// 应用模糊
applyBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    setStatus('loading', '正在应用模糊，请稍候...');
    previewBtn.disabled = true;
    applyBtn.disabled = true;
    
    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('mode', 'auto');
        formData.append('blur_type', blurType.value);
        formData.append('blur_strength', blurStrength.value);
        formData.append('text_prompt', textPromptInput.value || 'all objects');
        
        const response = await fetch(`${API_BASE}/v1/privacy/filter`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`请求失败: ${response.status} - ${errText}`);
        }
        
        const data = await response.json();
        
        // 显示最终模糊结果
        resultPreview.src = data.filtered_image_base64;
        resultPreview.style.display = 'block';
        resultPlaceholder.style.display = 'none';
        
        // 显示区域信息
        const regionCount = data.applied_regions.length;
        regionsInfo.textContent = `已对 ${regionCount} 个区域应用模糊`;
        
        setStatus('success', '处理完成！');
    } catch (err) {
        console.error(err);
        setStatus('error', `错误: ${err.message}`);
    } finally {
        previewBtn.disabled = !selectedFile;
        applyBtn.disabled = !selectedFile;  // 保持可用
    }
});

function setStatus(type, message) {
    statusDiv.style.display = 'block';
    statusDiv.className = `status ${type}`;
    statusDiv.textContent = message;
}
