/**
 * SAM3 隐私过滤前端逻辑
 */

const API_BASE = '';  // 同源，无需前缀

// DOM 元素
const imageInput = document.getElementById('image-input');
const blurType = document.getElementById('blur-type');
const blurStrength = document.getElementById('blur-strength');
const strengthValue = document.getElementById('strength-value');
const submitBtn = document.getElementById('submit-btn');
const statusDiv = document.getElementById('status');

const originalPlaceholder = document.getElementById('original-placeholder');
const originalPreview = document.getElementById('original-preview');
const resultPlaceholder = document.getElementById('result-placeholder');
const resultPreview = document.getElementById('result-preview');
const regionsInfo = document.getElementById('regions-info');

let selectedFile = null;

// 更新强度显示
blurStrength.addEventListener('input', () => {
    strengthValue.textContent = blurStrength.value;
});

// 图片选择
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) {
        selectedFile = null;
        submitBtn.disabled = true;
        originalPreview.style.display = 'none';
        originalPlaceholder.style.display = 'flex';
        return;
    }
    
    selectedFile = file;
    submitBtn.disabled = false;
    
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
});

// 提交处理
submitBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // 显示加载状态
    setStatus('loading', '处理中，请稍候...');
    submitBtn.disabled = true;
    
    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('mode', 'auto');
        formData.append('blur_type', blurType.value);
        formData.append('blur_strength', blurStrength.value);
        
        const response = await fetch(`${API_BASE}/v1/privacy/filter`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`请求失败: ${response.status} - ${errText}`);
        }
        
        const data = await response.json();
        
        // 显示结果
        resultPreview.src = data.filtered_image_base64;
        resultPreview.style.display = 'block';
        resultPlaceholder.style.display = 'none';
        
        // 显示区域信息
        const regionCount = data.applied_regions.length;
        regionsInfo.textContent = `已处理 ${regionCount} 个区域`;
        
        setStatus('success', '处理完成！');
    } catch (err) {
        console.error(err);
        setStatus('error', `错误: ${err.message}`);
    } finally {
        submitBtn.disabled = false;
    }
});

function setStatus(type, message) {
    statusDiv.style.display = 'block';
    statusDiv.className = `status ${type}`;
    statusDiv.textContent = message;
}
