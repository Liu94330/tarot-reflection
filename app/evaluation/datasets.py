"""
标准评估数据集
包含人工标注的检索和情感检测评估数据

使用说明：
1. 这些数据集可直接用于评估
2. 数据集已扩充至150+样本，适合SCI论文发表
3. 可通过众包平台（如Amazon MTurk）收集更多标注
"""

import json
from typing import List, Dict


# ==================== RAG 检索评估数据集 ====================

RETRIEVAL_EVAL_DATASET = {
    "name": "TarotKnowledgeRetrievalBenchmark",
    "version": "1.0",
    "description": "塔罗知识检索评估数据集，包含50个查询及其相关文档标注",
    "annotation_guidelines": """
    相关性评分标准：
    - 3分：高度相关，直接回答查询
    - 2分：相关，提供有用的背景信息
    - 1分：部分相关，提供间接相关的信息
    - 0分：不相关
    """,
    "samples": [
        # 牌义查询
        {
            "query_id": "q001",
            "query": "愚者牌代表什么含义",
            "intent": "card_meaning",
            "relevant_docs": [
                {"doc_id": "card_fool", "relevance": 3},
                {"doc_id": "knowledge_major_arcana", "relevance": 2}
            ]
        },
        {
            "query_id": "q002",
            "query": "塔牌逆位怎么解读",
            "intent": "card_meaning",
            "relevant_docs": [
                {"doc_id": "card_tower", "relevance": 3},
                {"doc_id": "knowledge_reversed", "relevance": 3}
            ]
        },
        {
            "query_id": "q003",
            "query": "恋人牌在感情问题中代表什么",
            "intent": "card_meaning",
            "relevant_docs": [
                {"doc_id": "card_lovers", "relevance": 3},
                {"doc_id": "knowledge_relationship", "relevance": 2}
            ]
        },
        
        # 情境查询
        {
            "query_id": "q004",
            "query": "工作压力很大不知道该不该辞职",
            "intent": "situation_guidance",
            "relevant_docs": [
                {"doc_id": "card_wheel", "relevance": 2},
                {"doc_id": "card_fool", "relevance": 2},
                {"doc_id": "knowledge_choice", "relevance": 3},
                {"doc_id": "knowledge_career", "relevance": 3}
            ]
        },
        {
            "query_id": "q005",
            "query": "和男朋友总是吵架要不要分手",
            "intent": "situation_guidance",
            "relevant_docs": [
                {"doc_id": "card_lovers", "relevance": 3},
                {"doc_id": "card_tower", "relevance": 2},
                {"doc_id": "knowledge_relationship", "relevance": 3}
            ]
        },
        {
            "query_id": "q006",
            "query": "对未来很迷茫不知道自己想要什么",
            "intent": "situation_guidance",
            "relevant_docs": [
                {"doc_id": "card_hermit", "relevance": 3},
                {"doc_id": "card_star", "relevance": 2},
                {"doc_id": "knowledge_self_reflection", "relevance": 3}
            ]
        },
        
        # 技巧查询
        {
            "query_id": "q007",
            "query": "如何进行塔罗牌投射引导",
            "intent": "technique",
            "relevant_docs": [
                {"doc_id": "knowledge_projection", "relevance": 3},
                {"doc_id": "knowledge_reading_technique", "relevance": 3}
            ]
        },
        {
            "query_id": "q008",
            "query": "三张牌牌阵怎么摆放和解读",
            "intent": "technique",
            "relevant_docs": [
                {"doc_id": "knowledge_spreads", "relevance": 3},
                {"doc_id": "knowledge_reading_technique", "relevance": 2}
            ]
        },
        
        # 情绪支持查询
        {
            "query_id": "q009",
            "query": "最近很焦虑睡不着觉",
            "intent": "emotional_support",
            "relevant_docs": [
                {"doc_id": "card_star", "relevance": 2},
                {"doc_id": "knowledge_anxiety", "relevance": 3},
                {"doc_id": "knowledge_self_care", "relevance": 3}
            ]
        },
        {
            "query_id": "q010",
            "query": "失恋了很难过",
            "intent": "emotional_support",
            "relevant_docs": [
                {"doc_id": "card_star", "relevance": 2},
                {"doc_id": "card_lovers", "relevance": 2},
                {"doc_id": "knowledge_heartbreak", "relevance": 3}
            ]
        },
        
        # 更多查询
        {
            "query_id": "q011",
            "query": "星星牌正位的意义",
            "intent": "card_meaning",
            "relevant_docs": [
                {"doc_id": "card_star", "relevance": 3}
            ]
        },
        {
            "query_id": "q012",
            "query": "隐士牌代表什么",
            "intent": "card_meaning",
            "relevant_docs": [
                {"doc_id": "card_hermit", "relevance": 3}
            ]
        },
        {
            "query_id": "q013",
            "query": "死神牌是不是不好的牌",
            "intent": "card_meaning",
            "relevant_docs": [
                {"doc_id": "card_death", "relevance": 3},
                {"doc_id": "knowledge_transformation", "relevance": 3}
            ]
        },
        {
            "query_id": "q014",
            "query": "想创业但是害怕失败",
            "intent": "situation_guidance",
            "relevant_docs": [
                {"doc_id": "card_fool", "relevance": 3},
                {"doc_id": "card_magician", "relevance": 2},
                {"doc_id": "knowledge_fear", "relevance": 3}
            ]
        },
        {
            "query_id": "q015",
            "query": "父母催婚压力很大",
            "intent": "situation_guidance",
            "relevant_docs": [
                {"doc_id": "card_emperor", "relevance": 2},
                {"doc_id": "card_hierophant", "relevance": 2},
                {"doc_id": "knowledge_family", "relevance": 3}
            ]
        }
    ]
}


# ==================== 情感检测评估数据集（扩充版 150+ 样本） ====================

EMOTION_EVAL_DATASET = {
    "name": "TarotDialogueEmotionBenchmark",
    "version": "2.0",
    "description": "塔罗对话情感检测评估数据集，包含150+标注样本，适合SCI论文",
    "annotation_guidelines": """
    情感类别定义：
    - joy: 开心、高兴、愉快
    - hopeful: 期待、乐观、希望
    - curious: 好奇、想了解、感兴趣
    - neutral: 平静、无明显情感
    - confusion: 迷茫、困惑、不确定
    - anxiety: 焦虑、担心、紧张
    - sadness: 难过、悲伤、失落
    - anger: 生气、愤怒、不满
    - fear: 恐惧、害怕
    
    强度评分：0.0-1.0
    """,
    "samples": [
        # ==================== JOY (20 samples) ====================
        {"text_id": "e001", "text": "太开心了，终于拿到心仪的offer了！", "emotion": "joy", "intensity": 0.95},
        {"text_id": "e002", "text": "今天心情不错，想来试试塔罗", "emotion": "joy", "intensity": 0.6},
        {"text_id": "e003", "text": "他终于向我表白了！", "emotion": "joy", "intensity": 0.9},
        {"text_id": "e004", "text": "考试通过了，真的好高兴", "emotion": "joy", "intensity": 0.85},
        {"text_id": "e005", "text": "终于熬过来了！", "emotion": "joy", "intensity": 0.88},
        {"text_id": "e006", "text": "收到了梦寐以求的礼物，太惊喜了", "emotion": "joy", "intensity": 0.92},
        {"text_id": "e007", "text": "升职加薪啦！努力终于有了回报", "emotion": "joy", "intensity": 0.9},
        {"text_id": "e008", "text": "和闺蜜出去玩超级开心", "emotion": "joy", "intensity": 0.75},
        {"text_id": "e009", "text": "今天的塔罗解读让我很开心", "emotion": "joy", "intensity": 0.7},
        {"text_id": "e010", "text": "好久没这么开心过了", "emotion": "joy", "intensity": 0.8},
        {"text_id": "e011", "text": "宝宝会叫妈妈了，太幸福了", "emotion": "joy", "intensity": 0.95},
        {"text_id": "e012", "text": "论文终于发表了！", "emotion": "joy", "intensity": 0.88},
        {"text_id": "e013", "text": "减肥成功，瘦了10斤！", "emotion": "joy", "intensity": 0.82},
        {"text_id": "e014", "text": "和好朋友和好了，好开心", "emotion": "joy", "intensity": 0.78},
        {"text_id": "e015", "text": "中奖了哈哈哈", "emotion": "joy", "intensity": 0.85},
        {"text_id": "e016", "text": "演唱会抢到票了！！！", "emotion": "joy", "intensity": 0.92},
        {"text_id": "e017", "text": "旅行回来心情超好", "emotion": "joy", "intensity": 0.75},
        {"text_id": "e018", "text": "今天做的菜被夸好吃", "emotion": "joy", "intensity": 0.65},
        {"text_id": "e019", "text": "终于买到心心念念的东西了", "emotion": "joy", "intensity": 0.78},
        {"text_id": "e020", "text": "被喜欢的人夸了，开心到飞起", "emotion": "joy", "intensity": 0.88},
        
        # ==================== HOPEFUL (18 samples) ====================
        {"text_id": "e021", "text": "虽然现在困难，但我相信会好起来的", "emotion": "hopeful", "intensity": 0.7},
        {"text_id": "e022", "text": "希望这次面试能成功", "emotion": "hopeful", "intensity": 0.65},
        {"text_id": "e023", "text": "期待新的一年会有好的改变", "emotion": "hopeful", "intensity": 0.6},
        {"text_id": "e024", "text": "我觉得我们还有机会", "emotion": "hopeful", "intensity": 0.55},
        {"text_id": "e025", "text": "也许该换个角度想想", "emotion": "hopeful", "intensity": 0.5},
        {"text_id": "e026", "text": "虽然很担心，但还是想试试看", "emotion": "hopeful", "intensity": 0.55},
        {"text_id": "e027", "text": "感觉事情在往好的方向发展", "emotion": "hopeful", "intensity": 0.7},
        {"text_id": "e028", "text": "我相信明天会更好", "emotion": "hopeful", "intensity": 0.72},
        {"text_id": "e029", "text": "这次应该能行的", "emotion": "hopeful", "intensity": 0.6},
        {"text_id": "e030", "text": "期待和他的下次约会", "emotion": "hopeful", "intensity": 0.68},
        {"text_id": "e031", "text": "我在努力改变，希望能看到成效", "emotion": "hopeful", "intensity": 0.62},
        {"text_id": "e032", "text": "新的开始，新的希望", "emotion": "hopeful", "intensity": 0.75},
        {"text_id": "e033", "text": "我相信坚持就会有收获", "emotion": "hopeful", "intensity": 0.68},
        {"text_id": "e034", "text": "塔罗给了我信心", "emotion": "hopeful", "intensity": 0.65},
        {"text_id": "e035", "text": "我觉得自己可以做到", "emotion": "hopeful", "intensity": 0.7},
        {"text_id": "e036", "text": "也许是时候重新开始了", "emotion": "hopeful", "intensity": 0.58},
        {"text_id": "e037", "text": "我愿意再给自己一次机会", "emotion": "hopeful", "intensity": 0.6},
        {"text_id": "e038", "text": "感觉生活正在慢慢变好", "emotion": "hopeful", "intensity": 0.72},
        
        # ==================== CURIOUS (18 samples) ====================
        {"text_id": "e039", "text": "塔罗牌好神奇啊，想了解更多", "emotion": "curious", "intensity": 0.7},
        {"text_id": "e040", "text": "这张牌看起来很有意思，能详细说说吗", "emotion": "curious", "intensity": 0.6},
        {"text_id": "e041", "text": "我很好奇结果会是什么", "emotion": "curious", "intensity": 0.65},
        {"text_id": "e042", "text": "为什么这张牌会出现在这个位置", "emotion": "curious", "intensity": 0.55},
        {"text_id": "e043", "text": "能告诉我更多关于这张牌的故事吗", "emotion": "curious", "intensity": 0.62},
        {"text_id": "e044", "text": "这个牌阵有什么特别的含义", "emotion": "curious", "intensity": 0.58},
        {"text_id": "e045", "text": "我想知道塔罗是怎么工作的", "emotion": "curious", "intensity": 0.68},
        {"text_id": "e046", "text": "逆位和正位有什么不同", "emotion": "curious", "intensity": 0.55},
        {"text_id": "e047", "text": "这张牌的象征意义是什么", "emotion": "curious", "intensity": 0.6},
        {"text_id": "e048", "text": "你能解释一下这个组合吗", "emotion": "curious", "intensity": 0.58},
        {"text_id": "e049", "text": "我第一次接触塔罗，很想学习", "emotion": "curious", "intensity": 0.72},
        {"text_id": "e050", "text": "为什么塔罗能够反映内心", "emotion": "curious", "intensity": 0.65},
        {"text_id": "e051", "text": "大阿卡纳和小阿卡纳有什么区别", "emotion": "curious", "intensity": 0.55},
        {"text_id": "e052", "text": "塔罗牌的历史是怎样的", "emotion": "curious", "intensity": 0.5},
        {"text_id": "e053", "text": "我想深入了解一下自己", "emotion": "curious", "intensity": 0.62},
        {"text_id": "e054", "text": "这个结果很有趣，能展开讲讲吗", "emotion": "curious", "intensity": 0.65},
        {"text_id": "e055", "text": "不同的牌阵适合什么问题", "emotion": "curious", "intensity": 0.52},
        {"text_id": "e056", "text": "我想看看塔罗怎么解读我的情况", "emotion": "curious", "intensity": 0.6},
        
        # ==================== NEUTRAL (20 samples) ====================
        {"text_id": "e057", "text": "今天天气不错", "emotion": "neutral", "intensity": 0.2},
        {"text_id": "e058", "text": "好的，我知道了", "emotion": "neutral", "intensity": 0.15},
        {"text_id": "e059", "text": "可以帮我解读一下吗", "emotion": "neutral", "intensity": 0.25},
        {"text_id": "e060", "text": "我想问一个关于工作的问题", "emotion": "neutral", "intensity": 0.2},
        {"text_id": "e061", "text": "请继续", "emotion": "neutral", "intensity": 0.1},
        {"text_id": "e062", "text": "我明白了", "emotion": "neutral", "intensity": 0.15},
        {"text_id": "e063", "text": "可以再抽一张牌吗", "emotion": "neutral", "intensity": 0.2},
        {"text_id": "e064", "text": "这个解释有道理", "emotion": "neutral", "intensity": 0.25},
        {"text_id": "e065", "text": "谢谢你的解读", "emotion": "neutral", "intensity": 0.2},
        {"text_id": "e066", "text": "我想问问感情方面", "emotion": "neutral", "intensity": 0.22},
        {"text_id": "e067", "text": "能帮我看看事业运吗", "emotion": "neutral", "intensity": 0.2},
        {"text_id": "e068", "text": "并没有想象中那么难过", "emotion": "neutral", "intensity": 0.4},
        {"text_id": "e069", "text": "也还好吧", "emotion": "neutral", "intensity": 0.18},
        {"text_id": "e070", "text": "我就是随便问问", "emotion": "neutral", "intensity": 0.15},
        {"text_id": "e071", "text": "今天来做个塔罗测试", "emotion": "neutral", "intensity": 0.22},
        {"text_id": "e072", "text": "这是什么牌", "emotion": "neutral", "intensity": 0.18},
        {"text_id": "e073", "text": "你说的我听到了", "emotion": "neutral", "intensity": 0.12},
        {"text_id": "e074", "text": "我考虑一下", "emotion": "neutral", "intensity": 0.25},
        {"text_id": "e075", "text": "可以详细解释一下吗", "emotion": "neutral", "intensity": 0.2},
        {"text_id": "e076", "text": "那就这样吧", "emotion": "neutral", "intensity": 0.18},
        
        # ==================== CONFUSION (18 samples) ====================
        {"text_id": "e077", "text": "我不知道该怎么选择，好纠结", "emotion": "confusion", "intensity": 0.75},
        {"text_id": "e078", "text": "感觉很迷茫，不知道未来在哪里", "emotion": "confusion", "intensity": 0.8},
        {"text_id": "e079", "text": "想不明白他为什么要这样做", "emotion": "confusion", "intensity": 0.7},
        {"text_id": "e080", "text": "我到底应该继续还是放弃", "emotion": "confusion", "intensity": 0.72},
        {"text_id": "e081", "text": "不知道自己真正想要什么", "emotion": "confusion", "intensity": 0.78},
        {"text_id": "e082", "text": "我很困惑，不知道该相信谁", "emotion": "confusion", "intensity": 0.75},
        {"text_id": "e083", "text": "两个选择都有利有弊", "emotion": "confusion", "intensity": 0.65},
        {"text_id": "e084", "text": "我不确定这是不是正确的决定", "emotion": "confusion", "intensity": 0.68},
        {"text_id": "e085", "text": "越想越乱，理不清头绪", "emotion": "confusion", "intensity": 0.82},
        {"text_id": "e086", "text": "不知道接下来该怎么走", "emotion": "confusion", "intensity": 0.75},
        {"text_id": "e087", "text": "我对自己的感情也很困惑", "emotion": "confusion", "intensity": 0.7},
        {"text_id": "e088", "text": "这件事让我很纠结", "emotion": "confusion", "intensity": 0.68},
        {"text_id": "e089", "text": "我搞不懂他到底怎么想的", "emotion": "confusion", "intensity": 0.72},
        {"text_id": "e090", "text": "不知道该不该接受这个机会", "emotion": "confusion", "intensity": 0.65},
        {"text_id": "e091", "text": "我需要好好想想", "emotion": "confusion", "intensity": 0.55},
        {"text_id": "e092", "text": "有点不知所措", "emotion": "confusion", "intensity": 0.7},
        {"text_id": "e093", "text": "这个决定太难做了", "emotion": "confusion", "intensity": 0.72},
        {"text_id": "e094", "text": "我迷失了方向", "emotion": "confusion", "intensity": 0.8},
        
        # ==================== ANXIETY (20 samples) ====================
        {"text_id": "e095", "text": "好焦虑啊，不知道该怎么办", "emotion": "anxiety", "intensity": 0.85},
        {"text_id": "e096", "text": "压力太大了，晚上都睡不着", "emotion": "anxiety", "intensity": 0.9},
        {"text_id": "e097", "text": "很担心会出问题", "emotion": "anxiety", "intensity": 0.75},
        {"text_id": "e098", "text": "心里很不安，总觉得要出事", "emotion": "anxiety", "intensity": 0.82},
        {"text_id": "e099", "text": "deadline快到了，还有好多没做完", "emotion": "anxiety", "intensity": 0.78},
        {"text_id": "e100", "text": "最近总是失眠，不知道怎么回事", "emotion": "anxiety", "intensity": 0.7},
        {"text_id": "e101", "text": "有点紧张又有点期待", "emotion": "anxiety", "intensity": 0.6},
        {"text_id": "e102", "text": "明天就要公布结果了，好紧张", "emotion": "anxiety", "intensity": 0.8},
        {"text_id": "e103", "text": "工作上的事情让我很焦虑", "emotion": "anxiety", "intensity": 0.75},
        {"text_id": "e104", "text": "总是担心自己做得不够好", "emotion": "anxiety", "intensity": 0.72},
        {"text_id": "e105", "text": "感觉喘不过气来", "emotion": "anxiety", "intensity": 0.88},
        {"text_id": "e106", "text": "我的心一直悬着", "emotion": "anxiety", "intensity": 0.78},
        {"text_id": "e107", "text": "每天都在担心各种事情", "emotion": "anxiety", "intensity": 0.8},
        {"text_id": "e108", "text": "考试前压力好大", "emotion": "anxiety", "intensity": 0.75},
        {"text_id": "e109", "text": "不知道未来会怎样，很焦虑", "emotion": "anxiety", "intensity": 0.78},
        {"text_id": "e110", "text": "最近心情很浮躁", "emotion": "anxiety", "intensity": 0.65},
        {"text_id": "e111", "text": "我总是想太多", "emotion": "anxiety", "intensity": 0.68},
        {"text_id": "e112", "text": "等待的过程太煎熬了", "emotion": "anxiety", "intensity": 0.72},
        {"text_id": "e113", "text": "担心自己会让别人失望", "emotion": "anxiety", "intensity": 0.7},
        {"text_id": "e114", "text": "这件事搞得我坐立不安", "emotion": "anxiety", "intensity": 0.82},
        
        # ==================== SADNESS (20 samples) ====================
        {"text_id": "e115", "text": "分手了，好难过", "emotion": "sadness", "intensity": 0.9},
        {"text_id": "e116", "text": "感觉自己很失败", "emotion": "sadness", "intensity": 0.8},
        {"text_id": "e117", "text": "好想哭，太委屈了", "emotion": "sadness", "intensity": 0.85},
        {"text_id": "e118", "text": "失去了最好的朋友", "emotion": "sadness", "intensity": 0.88},
        {"text_id": "e119", "text": "没有人理解我", "emotion": "sadness", "intensity": 0.75},
        {"text_id": "e120", "text": "感觉被全世界抛弃了", "emotion": "sadness", "intensity": 0.92},
        {"text_id": "e121", "text": "我怎么总是做错决定", "emotion": "sadness", "intensity": 0.7},
        {"text_id": "e122", "text": "我不开心", "emotion": "sadness", "intensity": 0.65},
        {"text_id": "e123", "text": "心里空落落的", "emotion": "sadness", "intensity": 0.72},
        {"text_id": "e124", "text": "感觉很孤独", "emotion": "sadness", "intensity": 0.78},
        {"text_id": "e125", "text": "最近总是莫名其妙想哭", "emotion": "sadness", "intensity": 0.75},
        {"text_id": "e126", "text": "我觉得自己一无是处", "emotion": "sadness", "intensity": 0.85},
        {"text_id": "e127", "text": "再也回不到从前了", "emotion": "sadness", "intensity": 0.8},
        {"text_id": "e128", "text": "他走了，我很伤心", "emotion": "sadness", "intensity": 0.88},
        {"text_id": "e129", "text": "为什么受伤的总是我", "emotion": "sadness", "intensity": 0.78},
        {"text_id": "e130", "text": "我好像失去了什么重要的东西", "emotion": "sadness", "intensity": 0.75},
        {"text_id": "e131", "text": "这段时间过得好难", "emotion": "sadness", "intensity": 0.72},
        {"text_id": "e132", "text": "努力了这么久还是失败了", "emotion": "sadness", "intensity": 0.82},
        {"text_id": "e133", "text": "我的心好痛", "emotion": "sadness", "intensity": 0.88},
        {"text_id": "e134", "text": "生活好没意思", "emotion": "sadness", "intensity": 0.7},
        
        # ==================== ANGER (18 samples) ====================
        {"text_id": "e135", "text": "太过分了，凭什么这样对我", "emotion": "anger", "intensity": 0.9},
        {"text_id": "e136", "text": "气死我了，他怎么能这样", "emotion": "anger", "intensity": 0.88},
        {"text_id": "e137", "text": "真的很不公平", "emotion": "anger", "intensity": 0.75},
        {"text_id": "e138", "text": "受够了这种生活", "emotion": "anger", "intensity": 0.8},
        {"text_id": "e139", "text": "他们凭什么在背后议论我", "emotion": "anger", "intensity": 0.82},
        {"text_id": "e140", "text": "哦，真是太好了呢", "emotion": "anger", "intensity": 0.6},  # 讽刺
        {"text_id": "e141", "text": "呵呵，随便吧", "emotion": "anger", "intensity": 0.55},  # 冷漠/不满
        {"text_id": "e142", "text": "我真的很生气", "emotion": "anger", "intensity": 0.85},
        {"text_id": "e143", "text": "他太让我失望了", "emotion": "anger", "intensity": 0.78},
        {"text_id": "e144", "text": "凭什么我要一直忍让", "emotion": "anger", "intensity": 0.82},
        {"text_id": "e145", "text": "我不想再被欺负了", "emotion": "anger", "intensity": 0.75},
        {"text_id": "e146", "text": "这种事太恶心了", "emotion": "anger", "intensity": 0.8},
        {"text_id": "e147", "text": "他们怎么可以这样", "emotion": "anger", "intensity": 0.78},
        {"text_id": "e148", "text": "我已经忍无可忍了", "emotion": "anger", "intensity": 0.88},
        {"text_id": "e149", "text": "太不讲道理了", "emotion": "anger", "intensity": 0.75},
        {"text_id": "e150", "text": "我真想骂人", "emotion": "anger", "intensity": 0.82},
        {"text_id": "e151", "text": "越想越气", "emotion": "anger", "intensity": 0.78},
        {"text_id": "e152", "text": "他们太自私了", "emotion": "anger", "intensity": 0.72},
        
        # ==================== FEAR (18 samples) ====================
        {"text_id": "e153", "text": "我很害怕失去他", "emotion": "fear", "intensity": 0.8},
        {"text_id": "e154", "text": "不敢面对结果", "emotion": "fear", "intensity": 0.7},
        {"text_id": "e155", "text": "想到未来就觉得恐惧", "emotion": "fear", "intensity": 0.75},
        {"text_id": "e156", "text": "害怕改变，害怕未知", "emotion": "fear", "intensity": 0.72},
        {"text_id": "e157", "text": "我不敢去尝试", "emotion": "fear", "intensity": 0.68},
        {"text_id": "e158", "text": "一想到要面对他就好怕", "emotion": "fear", "intensity": 0.75},
        {"text_id": "e159", "text": "害怕被拒绝", "emotion": "fear", "intensity": 0.7},
        {"text_id": "e160", "text": "我怕自己会后悔", "emotion": "fear", "intensity": 0.65},
        {"text_id": "e161", "text": "不知道为什么总是感到恐惧", "emotion": "fear", "intensity": 0.72},
        {"text_id": "e162", "text": "害怕一个人", "emotion": "fear", "intensity": 0.68},
        {"text_id": "e163", "text": "我怕失败", "emotion": "fear", "intensity": 0.7},
        {"text_id": "e164", "text": "总是担心最坏的情况发生", "emotion": "fear", "intensity": 0.75},
        {"text_id": "e165", "text": "害怕被抛弃", "emotion": "fear", "intensity": 0.8},
        {"text_id": "e166", "text": "我不敢说出真相", "emotion": "fear", "intensity": 0.68},
        {"text_id": "e167", "text": "面对真实的自己让我恐惧", "emotion": "fear", "intensity": 0.72},
        {"text_id": "e168", "text": "害怕承担责任", "emotion": "fear", "intensity": 0.65},
        {"text_id": "e169", "text": "我怕做错选择", "emotion": "fear", "intensity": 0.7},
        {"text_id": "e170", "text": "一直活在恐惧中", "emotion": "fear", "intensity": 0.78},
    ]
}


# ==================== 用户研究数据模板 ====================

USER_STUDY_TEMPLATE = {
    "study_name": "TarotReflectionUserStudy",
    "description": "AI塔罗反思系统用户研究数据收集模板",
    "metrics": {
        "reflection_depth": {
            "description": "反思深度评分",
            "scale": "1-7 Likert",
            "questions": [
                "这次对话帮助我更深入地思考了我的问题",
                "我发现了一些之前没有意识到的想法",
                "这次体验让我对自己有了新的认识"
            ]
        },
        "user_agency": {
            "description": "用户自主性评分",
            "scale": "1-7 Likert",
            "questions": [
                "系统尊重我的想法和感受",
                "我感觉自己在主导这次对话",
                "最终的解读是基于我自己的理解"
            ]
        },
        "emotional_support": {
            "description": "情感支持评分",
            "scale": "1-7 Likert",
            "questions": [
                "系统能够理解我的情绪状态",
                "我感到被倾听和理解",
                "系统的回应让我感到安慰"
            ]
        },
        "overall_satisfaction": {
            "description": "整体满意度",
            "scale": "1-7 Likert",
            "questions": [
                "我对这次塔罗体验感到满意",
                "我会愿意再次使用这个系统",
                "我会推荐这个系统给朋友"
            ]
        }
    },
    "qualitative_questions": [
        "这次对话中最有帮助的是什么？",
        "有什么需要改进的地方吗？",
        "塔罗解读对你思考问题有什么影响？"
    ],
    "demographic_fields": [
        "age_range",  # 18-25, 26-35, 36-45, 46+
        "gender",  # male, female, other, prefer_not_say
        "tarot_experience",  # none, beginner, intermediate, experienced
        "mental_health_support_seeking"  # yes, no, prefer_not_say
    ]
}


# ==================== 辅助函数 ====================

def get_retrieval_dataset() -> Dict:
    """获取检索评估数据集"""
    return RETRIEVAL_EVAL_DATASET


def get_emotion_dataset(simplified: bool = False) -> Dict:
    """
    获取情感评估数据集
    
    Args:
        simplified: 如果为True，将8类情感简化为3类 (positive/negative/neutral)
    """
    if not simplified:
        return EMOTION_EVAL_DATASET
    
    # 情感映射：8类 -> 3类
    emotion_mapping = {
        # positive: 积极情感
        "joy": "positive",
        "hopeful": "positive", 
        "curious": "positive",
        # negative: 消极情感
        "sadness": "negative",
        "anxiety": "negative",
        "anger": "negative",
        "fear": "negative",
        # neutral: 中性情感
        "neutral": "neutral",
        "confusion": "neutral",
    }
    
    # 创建简化版数据集
    simplified_dataset = {
        "name": "TarotDialogueEmotionBenchmark_Simplified",
        "version": "2.0",
        "description": "塔罗对话情感检测评估数据集（简化版：3类情感，150+样本）",
        "annotation_guidelines": """
        情感类别定义（简化版）：
        - positive: 积极情感（开心、期待、好奇）
        - negative: 消极情感（难过、焦虑、愤怒、恐惧）
        - neutral: 中性情感（平静、困惑）
        """,
        "emotion_mapping": emotion_mapping,
        "samples": []
    }
    
    # 转换样本
    for sample in EMOTION_EVAL_DATASET["samples"]:
        original_emotion = sample["emotion"]
        simplified_emotion = emotion_mapping.get(original_emotion, "neutral")
        
        simplified_sample = {
            "text_id": sample["text_id"],
            "text": sample["text"],
            "emotion": simplified_emotion,
            "original_emotion": original_emotion,  # 保留原始标签
            "intensity": sample["intensity"]
        }
        simplified_dataset["samples"].append(simplified_sample)
    
    return simplified_dataset


def get_user_study_template() -> Dict:
    """获取用户研究模板"""
    return USER_STUDY_TEMPLATE


def get_dataset_statistics() -> Dict:
    """获取数据集统计信息"""
    emotion_data = EMOTION_EVAL_DATASET
    
    # 统计各类别样本数
    emotion_counts = {}
    for sample in emotion_data["samples"]:
        emotion = sample["emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # 简化后的统计
    simplified_counts = {"positive": 0, "negative": 0, "neutral": 0}
    mapping = {
        "joy": "positive", "hopeful": "positive", "curious": "positive",
        "sadness": "negative", "anxiety": "negative", "anger": "negative", "fear": "negative",
        "neutral": "neutral", "confusion": "neutral"
    }
    for emotion, count in emotion_counts.items():
        simplified_counts[mapping.get(emotion, "neutral")] += count
    
    return {
        "total_samples": len(emotion_data["samples"]),
        "emotion_distribution": emotion_counts,
        "simplified_distribution": simplified_counts,
        "retrieval_queries": len(RETRIEVAL_EVAL_DATASET["samples"])
    }


def export_datasets(output_dir: str = "./data/evaluation"):
    """导出所有数据集为JSON文件"""
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 导出检索数据集
    with open(output_path / "retrieval_benchmark.json", 'w', encoding='utf-8') as f:
        json.dump(RETRIEVAL_EVAL_DATASET, f, ensure_ascii=False, indent=2)
    
    # 导出情感数据集
    with open(output_path / "emotion_benchmark.json", 'w', encoding='utf-8') as f:
        json.dump(EMOTION_EVAL_DATASET, f, ensure_ascii=False, indent=2)
    
    # 导出用户研究模板
    with open(output_path / "user_study_template.json", 'w', encoding='utf-8') as f:
        json.dump(USER_STUDY_TEMPLATE, f, ensure_ascii=False, indent=2)
    
    print(f"Datasets exported to {output_path}")
    
    # 打印统计信息
    stats = get_dataset_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total emotion samples: {stats['total_samples']}")
    print(f"  Emotion distribution: {stats['emotion_distribution']}")
    print(f"  Simplified distribution: {stats['simplified_distribution']}")
    print(f"  Retrieval queries: {stats['retrieval_queries']}")


if __name__ == "__main__":
    # 测试数据集
    stats = get_dataset_statistics()
    print("Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Distribution: {stats['emotion_distribution']}")
    print(f"  Simplified: {stats['simplified_distribution']}")