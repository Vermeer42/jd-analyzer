import streamlit as st
from openai import OpenAI
import easyocr
import json
import pandas as pd
from PIL import Image
import numpy as np
import io # 新增：用于在内存中生成 Excel 文件供下载

# 1. 页面配置
st.set_page_config(page_title="AI 智能求职分析引擎", page_icon="🚀", layout="wide")

# ================= 核心知识点：初始化状态管理 (Session State) =================
# 告诉网页：如果没有历史记录保险箱，就建一个空的列表
if 'history' not in st.session_state:
    st.session_state.history = []

# 2. 缓存大模型与视觉模型
@st.cache_resource
def load_models():
    # 让代码去系统的安全保险箱里找密码
    client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    reader = easyocr.Reader(['ch_sim', 'en'])
    return client, reader

client, reader = load_models()

# ================= 侧边栏：通用化配置参数 =================
with st.sidebar:
    st.header("👤 候选人画像配置")
    st.markdown("填入你的真实背景，AI 将基于此进行严苛匹配。")
    
    # 将写死的信息变成用户可填写的交互框，并提供默认值
    user_school = st.text_input("学校与学历", value="南京大学")
    user_major = st.text_input("专业", value="社会学硕士在读")
    user_exp = st.text_area("核心实习/项目经历", value="里斯战略咨询实习，精通深度访谈、消费者KBF提炼、案头研究与行业洞察。")
    user_target = st.text_input("期望岗位方向", value="用户研究、战略、咨询、产品经理")
    
    st.divider()
    st.info("💡 提示：你可以随时修改左侧的画像，新的画像将只应用于接下来分析的 JD，不会影响下方已有的历史记录。")

# ================= 动态生成 System Prompt =================
# 只有把用户输入嵌进去，AI 才知道当前在给谁做咨询
DYNAMIC_PROFILE = f"""
# Candidate Profile (必须牢记)
- 学校与学历：{user_school}
- 专业：{user_major}
- 核心经历与技能：{user_exp}
- 职业方向：{user_target}。只要 JD 能发挥上述技能，即视为高匹配。
"""

# 注意：JSON 格式部分用另一个字符串拼接，防止被 f-string 的大括号影响
JSON_RULES = """
# Scoring Rules (基础分 50)
- 加分：与核心技能匹配度高(+15)；大厂/头部公司(+20)；一线及新一线城市(+10)；转正机会(+10)。
- 扣分：打杂/日常纯执行(-10)；隐性加班黑话(-10)；日薪低于行业基准(-15)。
- 否决：纯理工/代码背景且无产品思维、单休。

# Workflow
1. 清洗乱码，理解 JD 的真实意图。
2. 内部进行思维链推理。
3. 强制输出纯 JSON 格式数据。

# JSON 强制结构
{
    "公司": "提取公司名",
    "公司业务简介": "用一两句话简述该公司的核心业务或行业地位",
    "岗位": "提取岗位名",
    "Base地点": "城市",
    "薪资待遇": "例如：250元/天",
    "出勤要求": "例如：每周4天",
    "硬技能要求": "提取明确的技能，无则填无",
    "最终得分": 55,
    "加扣分明细": "加分：xxx(+15)；扣分：xxx(-20)",
    "核心短评": "20字以内一针见血的评价",
    "经验复用点": "说明候选人的经历能否在此复用，具体是哪一点",
    "转型卡点": "说明缺乏什么具体经验或硬技能",
    "综合详细建议": "50-100字详细分析。说明归因及面试建议",
    "JD核心内容精简": "将原JD去粗取精，按岗位职责和任职要求列出"
}
"""

SYSTEM_PROMPT = DYNAMIC_PROFILE + JSON_RULES

# ================= 主体 UI =================
st.title("🚀 AI 智能求职分析与匹配引擎")
st.markdown("上传 JD 截图，秒级输出**多维度匹配报告**。支持多岗位连续分析与横向对比。")

# 1. 顶部操作区
uploaded_file = st.file_uploader("📸 点击或拖拽上传 JD 截图 (支持 png/jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 修复了你截图中提到的黄色警告 (使用 use_container_width)
    image = Image.open(uploaded_file)
    st.image(image, caption="待分析的 JD 截图", use_container_width=True)
    
    # 突出显示的按钮：加上 type="primary" 会变成醒目的主题色
    if st.button("⚡ 开始深度分析 (提取文字并呼叫大脑)", type="primary", use_container_width=True):
        with st.spinner("🔍 视觉模块正在识字，大脑正在高速运算，请稍候..."):
            try:
                # 视觉识字
                # --- 新增：图片极致压缩逻辑，防止云服务器爆内存 ---
                # 手机截图分辨率极高，强制将其宽度等比例缩小到 800 像素，OCR 依然能看清，但内存占用骤降 80%
                max_width = 800
                if image.width > max_width:
                    ratio = max_width / image.width
                    new_height = int(image.height * ratio)
                    # 使用 Image.Resampling.LANCZOS 保证缩小后文字依然清晰锐利
                    image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
                # ------------------------------------------------
                
                # 视觉识字 (使用瘦身后的图片)
                img_array = np.array(image.convert('RGB'))
                ocr_result = reader.readtext(img_array, detail=0)
                jd_content = "\n".join(ocr_result)
                
                # 大脑运算
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": jd_content}
                    ],
                    response_format={"type": "json_object"} 
                )
                jd_data = json.loads(response.choices[0].message.content)
                
                # 关键一步：把分析结果存入 Session 保险箱！
                st.session_state.history.append(jd_data)
                st.success(f"🎉 分析成功！【{jd_data['公司']} - {jd_data['岗位']}】已加入下方对比库。")
                
            except Exception as e:
                st.error(f"分析过程中遇到问题: {e}")

st.divider()

# ================= 历史记录展示与操作区 =================
if st.session_state.history:
    st.subheader(f"📊 岗位分析对比库 (共 {len(st.session_state.history)} 条记录)")
    
    # -- 核心功能 1：导出 Excel --
    # 把 Session 里的数据转成表格
    df_history = pd.DataFrame(st.session_state.history)
    
    # 在内存中把表格转成 Excel 文件字节流
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_history.to_excel(writer, index=False, sheet_name='岗位分析数据')
    excel_data = output.getvalue()
    
    # 提供下载按钮
    st.download_button(
        label="📥 导出当前所有数据至 Excel",
        data=excel_data,
        file_name="我的岗位数据库.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    st.write("") # 空行排版
    
# -- 核心功能 2：展示并支持删除 --
    # 倒序展示，让最新传的显示在最上面
    for i, record in enumerate(reversed(st.session_state.history)):
        # 算出它在原始列表里的真实索引
        real_index = len(st.session_state.history) - 1 - i 
        
        # 标题栏：展示最核心的 得分 + 公司 + 岗位
        title = f"⭐ 【{record.get('最终得分', '?')}分】 {record.get('公司', '未知')} · {record.get('岗位', '未知')}"
        
        with st.expander(title, expanded=(i==0)):
            # 第一层：公司简介（灰色小字，增加背景感）
            st.caption(f"🏢 公司概况：{record.get('公司业务简介', '暂无简介')}")
            
            # 第二层：左右分栏布局
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**🎯 核心短评：** {record.get('核心短评', '')}")
                
                # 用 info 组件突出展示加扣分明细，逻辑感更强
                st.info(f"⚖️ **计分逻辑：** {record.get('加扣分明细', '未记录')}")
                
                st.markdown(f"**💡 综合建议：** {record.get('综合详细建议', '')}")
                
                # 经验与卡点并排展示
                sub_col_a, sub_col_b = st.columns(2)
                with sub_col_a:
                    st.success(f"✅ **经验复用：**\n\n{record.get('经验复用点', '')}")
                with sub_col_b:
                    st.warning(f"⚠️ **转型卡点：**\n\n{record.get('转型卡点', '')}")
            
            with col2:
                # 右侧放结构化硬信息
                st.write(f"📍 **地点：** {record.get('Base地点', '未知')}")
                st.write(f"💰 **薪资：** {record.get('薪资待遇', '未知')}")
                st.write(f"📅 **出勤：** {record.get('出勤要求', '未知')}")
                st.write(f"🛠️ **硬技能：** {record.get('硬技能要求', '无')}")
                
                st.divider()
                # 删除按钮
                if st.button("🗑️ 丢弃此记录", key=f"del_{real_index}", use_container_width=True):
                    st.session_state.history.pop(real_index)
                    st.rerun()
            
            # 第三层：嵌套折叠 JD 内容精简（不占空间，想看才点开）
            with st.status("📄 查看清洗后的 JD 原文要求"):
                st.write(record.get('JD核心内容精简', '暂无内容'))