import torch
import json
from tqdm import tqdm
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# --- C1: 实现图像 Patch 打乱 (隔离视觉信息) ---
def patch_shuffle(image_tensor, num_patches=14):
    """
    将图像划分为 num_patches * num_patches 个块并随机打乱
    这是论文中提到的 C1 组件，用于提取语言偏见信号
    """
    c, h, w = image_tensor.shape
    p_h, p_w = h // num_patches, w // num_patches
    
    # 划分 patches
    patches = image_tensor.unfold(1, p_h, p_h).unfold(2, p_w, p_w)
    patches = patches.contiguous().view(c, -1, p_h, p_w)
    
    # 随机打乱
    idx = torch.randperm(patches.size(1))
    patches = patches[:, idx, :, :]
    
    # 重组图像
    shuffled = patches.view(c, num_patches, num_patches, p_h, p_w)
    shuffled = shuffled.permute(0, 1, 3, 2, 4).contiguous().view(c, h, w)
    return shuffled

# --- C2: 非对称对比解码逻辑 ---
def run_inference():
    model_path = "liuhaotian/llava-v1.5-7b" # 替换为你自己的模型路径
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava-v1.5-7b")
    
    # 假设加载 POPE 数据
    questions = [json.loads(q) for q in open("coco_pope_adversarial.jsonl", "r")]
    results = []

    for line in tqdm(questions):
        image = Image.open(f"val2014/{line['image']}").convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['images'].half().cuda()
        
        # 准备打乱后的图像 (C1)
        shuffled_tensor = patch_shuffle(image_tensor[0]).unsqueeze(0)
        
        input_ids = tokenizer_image_token(line['text'], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            # 获取原始预测
            logits_orig = model(input_ids, images=image_tensor).logits[:, -1, :]
            pred_id = torch.argmax(logits_orig, dim=-1)
            
            # 非对称惩罚 (C2): 只有当原始预测倾向于 "Yes" 时才进行对比增强
            # 这里的 3848 和 3878 需根据具体 Tokenizer 的 'Yes'/'No' 对应
            if pred_id == 3848: # 假设 3848 是 'Yes'
                logits_bias = model(input_ids, images=shuffled_tensor).logits[:, -1, :]
                # 惩罚偏见信号
                final_logits = (1 + 0.15) * logits_orig - 0.15 * logits_bias
                final_pred = torch.argmax(final_logits, dim=-1)
            else:
                final_pred = pred_id

        # 保存结果用于后续指标计算
        results.append({
            "question_id": line["question_id"],
            "model_answer": "Yes" if final_pred == 3848 else "No",
            "label": line["label"]
        })

    with open("results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    run_inference()
