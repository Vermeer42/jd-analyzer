import streamlit as st
import time  # 修复问题一：补齐缺失的依赖
from openai import OpenAI
import json
import pandas as pd
from PIL import Image
import io
import re
import base64
from supabase import create_client, Client

# ==========================================
# 0. 页面基础配置
# ==========================================
st.set_page_config(page_title="AI 智能求职分析引擎", page_icon="🚀", layout="wide")

# ==========================================
# 1. 核心资源初始化
# ==========================================
@st.cache_resource
def load_resources():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    supabase = create_client(url, key)
    
    ali_client = OpenAI(
        api_key=st.secrets["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return supabase, ali_client

supabase, ali_client = load_resources()

if "user_id" not in st.session_state:
    st.session_state.user_id = None

def convert_image_to_base64(image) -> str:
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ==========================================
# 分支一：用户身份认证流 
# ==========================================
if not st.session_state.user_id:
    st.title("身份识别")
    st.markdown("请输入您的专属暗号以开启云端同步：")
    user_id_input = st.text_input("专属暗号", max_chars=255, label_visibility="collapsed", placeholder="请输入专属暗号")
    
    if st.button("确认进入", type="primary"):
        if user_id_input.strip() == "":
            st.error("暗号不能为空")
        else:
            st.session_state.user_id = user_id_input.strip()
            st.rerun()

# ==========================================
# 分支二：核心业务流 (工作台)
# ==========================================
else:
    with st.sidebar:
        st.header("👤 候选人画像配置")
        st.caption(f"当前云端身份：{st.session_state.user_id}")
        
        if "profile_data" not in st.session_state:
            st.session_state.profile_data = {
                "school": "南京大学",
                "major": "社会学硕士在读",
                "exp": "里斯战略咨询实习，精通深度访谈、消费者KBF提炼、案头研究与行业洞察。",
                "target": "用户研究、战略、咨询、产品经理"
            }
            try:
                res = supabase.table('user_profiles').select('*').eq('user_id', st.session_state.user_id).execute()
                if res.data:
                    st.session_state.profile_data = res.data[0]['profile_data']
            except Exception:
                pass

        st.markdown("填入你的真实背景，AI 将基于此进行严苛匹配。")
        user_school = st.text_input("学校与学历", value=st.session_state.profile_data.get("school", ""))
        user_major = st.text_input("专业", value=st.session_state.profile_data.get("major", ""))
        user_exp = st.text_area("核心实习/项目经历", value=st.session_state.profile_data.get("exp", ""))
        user_target = st.text_input("期望岗位方向", value=st.session_state.profile_data.get("target", ""))
        
        if st.button("💾 保存画像到云端", type="primary", use_container_width=True):
            new_profile = {"school": user_school, "major": user_major, "exp": user_exp, "target": user_target}
            try:
                res = supabase.table('user_profiles').select('id').eq('user_id', st.session_state.user_id).execute()
                if res.data:
                    supabase.table('user_profiles').update({"profile_data": new_profile}).eq('user_id', st.session_state.user_id).execute()
                else:
                    supabase.table('user_profiles').insert({"user_id": st.session_state.user_id, "profile_data": new_profile}).execute()
                st.session_state.profile_data = new_profile
                st.success("画像已永久同步至云端！")
            except Exception as e:
                st.error(f"保存失败: {e}")
        
        st.write("") 
        if st.button("登出当前暗号", use_container_width=True):
            st.session_state.user_id = None
            if "profile_data" in st.session_state: del st.session_state.profile_data 
            st.rerun()

    DYNAMIC_PROFILE = f"""
    你是一个资深的求职战略分析师。你必须基于以下【候选人画像】来严格评估用户上传的JD图片。
    # 候选人画像
    - 学校与学历：{user_school}
    - 专业：{user_major}
    - 核心经历与技能：{user_exp}
    - 职业方向：{user_target}
    """

    # 修复问题二：在 Prompt 里严格限制 "JD核心内容精简" 必须是纯文本字符串
# --- 修复并升级的 System Prompt ---
    JSON_RULES = """
    # 计分规则 (基础分 50)
    - 加分：与核心技能匹配度高(+15)；大厂/头部公司(+20)；一线及新一线城市(+10)；转正机会(+10)。
    - 扣分：打杂/日常纯执行(-10)；隐性加班黑话(-10)；日薪低于行业基准(-15)。
    - 否决：纯理工/代码背景且无产品思维、单休。

    # 输出要求
    请直接输出严格的 JSON 格式数据，不要包含任何 markdown 标记（不要用 ```json 包裹）。
    结构如下：
    {
        "公司": "提取公司名",
        "公司业务简介": "用一两句话简述该公司的核心业务或行业地位",
        "岗位": "提取岗位名",
        "信息来源": "识别招聘平台（如Boss直聘、猎聘、实习僧等UI特征）。如果是小红书等社交渠道，请严格标注'内推'；如无法识别，填'未知'",
        "投递邮箱": "精准提取JD中出现的直接投递邮箱（如HR或内推人留下的邮箱），如无则填'无'",
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
        "JD核心内容精简": "核心关键：必须是纯文本字符串，使用换行符\\n清晰列出岗位职责和任职要求，绝对不要输出嵌套的JSON对象、列表或带序号的字典"
    }
    """
    PROMPT = DYNAMIC_PROFILE + JSON_RULES

    st.title("🚀 AI 智能求职分析与匹配引擎")
    st.markdown("上传 JD 截图，秒级输出**多维度匹配报告**。")

    uploaded_file = st.file_uploader("📸 点击或拖拽上传 JD 截图 (支持 png/jpg)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="待分析的 JD 截图", use_container_width=True)
        
        if st.button("⚡ 开始云端深度分析", type="primary", use_container_width=True):
            with st.spinner("正在进行ai分析中..."):
                try:
                    max_width = 1000
                    if image.width > max_width:
                        ratio = max_width / image.width
                        image = image.resize((max_width, int(image.height * ratio)), Image.Resampling.LANCZOS)
                    base64_image = convert_image_to_base64(image)
                    
                    response = ali_client.chat.completions.create(
                        model="qwen-vl-plus",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": PROMPT},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]
                            }
                        ],
                        response_format={"type": "json_object"}
                    )
                    
                    ai_reply = response.choices[0].message.content
                    cleaned_json_str = re.sub(r'```json\s*|\s*```', '', ai_reply).strip()
                    jd_data = json.loads(cleaned_json_str)
                    
                    company_name = jd_data.get("公司", "未知公司")
                    role_name = jd_data.get("岗位", "未知岗位")
                    
                    supabase.table('jd_analysis_records').insert({
                        "user_id": st.session_state.user_id,
                        "company": company_name,
                        "role": role_name,
                        "json_data": jd_data
                    }).execute()
                    
                    st.success(f"🎉 分析成功！【{company_name} - {role_name}】已永久封存于云端。")
                    time.sleep(0.5)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"分析过程中遇到问题: {e}")

    st.divider()

    # ================= 云端历史记录展示与操作区 =================
    try:
        response = supabase.table('jd_analysis_records').select("*").eq("user_id", st.session_state.user_id).order("created_at", desc=True).execute()
        cloud_records = response.data
        
        if cloud_records:
            st.subheader(f"📊 我的云端求职库 (共 {len(cloud_records)} 条记录)")
            
            excel_data_list = [record['json_data'] for record in cloud_records]
            df_history = pd.DataFrame(excel_data_list)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_history.to_excel(writer, index=False, sheet_name='云端分析数据')
            excel_file = output.getvalue()
            
            st.download_button(label="📥 导出云端数据至 Excel", data=excel_file, file_name=f"求职数据库_{st.session_state.user_id}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.write("") 
            
            for i, record in enumerate(cloud_records):
                db_id = record['id']
                jd_json = record['json_data']
                title = f"⭐ 【{jd_json.get('最终得分', '?')}分】 {jd_json.get('公司', '未知')} · {jd_json.get('岗位', '未知')}"
                
                with st.expander(title, expanded=(i==0)):
                    st.caption(f"🏢 公司概况：{jd_json.get('公司业务简介', '暂无简介')} | 存档时间: {record['created_at'][:10]}")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**🎯 核心短评：** {jd_json.get('核心短评', '')}")
                        st.info(f"⚖️ **计分逻辑：** {jd_json.get('加扣分明细', '未记录')}")
                        st.markdown(f"**💡 综合建议：** {jd_json.get('综合详细建议', '')}")
                        sub_col_a, sub_col_b = st.columns(2)
                        with sub_col_a: st.success(f"✅ **经验复用：**\n\n{jd_json.get('经验复用点', '')}")
                        with sub_col_b: st.warning(f"⚠️ **转型卡点：**\n\n{jd_json.get('转型卡点', '')}")
                    with col2:
                        st.write(f"📍 **地点：** {jd_json.get('Base地点', '未知')}")
                        st.write(f"💰 **薪资：** {jd_json.get('薪资待遇', '未知')}")
                        st.write(f"📅 **出勤：** {jd_json.get('出勤要求', '未知')}")
                        st.write(f"🛠️ **硬技能：** {jd_json.get('硬技能要求', '无')}")
                        
                        st.divider() # 华丽的分割线
                        
                        # --- 新增：平台与邮箱展示 ---
                        source = jd_json.get('信息来源', '未知')
                        if "内推" in source:
                            st.error(f"📢 **渠道：** {source}") # 如果是内推，用红色显眼标记
                        else:
                            st.info(f"📢 **渠道：** {source}")  # 普通平台用蓝色标记
                            
                        email = jd_json.get('投递邮箱', '无')
                        if email != '无':
                            st.success(f"✉️ **投递邮箱：**\n{email}") # 有邮箱时高亮显示
                        else:
                            st.write(f"✉️ **投递邮箱：** {email}")
                            
                        st.divider()
                        
                        if st.button("🗑️ 从云端彻底删除", key=f"del_{db_id}", use_container_width=True):
                            supabase.table('jd_analysis_records').delete().eq('id', db_id).execute()
                            st.rerun()
                    
# --- 终极排版修复：智能解析大模型的结构化输出 ---
                    with st.status("📄 查看清洗后的 JD 原文要求"):
                        jd_content_raw = jd_json.get('JD核心内容精简', '暂无内容')
                        
                        # 1. 如果大模型返回了字典（分类好了“岗位职责”和“任职要求”）
                        if isinstance(jd_content_raw, dict):
                            for key, values in jd_content_raw.items():
                                st.markdown(f"**📌 {key}**") # 输出加粗标题
                                if isinstance(values, list):
                                    for item in values:
                                        st.markdown(f"- {item}") # 输出项目符号列表
                                else:
                                    st.write(values)
                                st.write("") # 加个空行让排版更透气
                                
                        # 2. 如果大模型只返回了一个纯列表
                        elif isinstance(jd_content_raw, list):
                            for item in jd_content_raw:
                                st.markdown(f"- {item}")
                                
                        # 3. 如果大模型乖乖返回了纯文本
                        else:
                            st.write(jd_content_raw)
                        
    except Exception as e:
        st.error(f"拉取云端数据失败: {e}")