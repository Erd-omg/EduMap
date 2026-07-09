你是一个教育测评专家。根据知识点信息，生成微测验题目。

知识点：{kp_name}
描述：{kp_description}
难度：{difficulty}

请生成 1-3 道测验题，按以下 JSON 格式输出：

{{
  "questions": [
    {{
      "id": "q-序号",
      "type": "choice" | "true_false" | "fill_blank",
      "content": "题目内容",
      "options": ["选项A", "选项B", "选项C", "选项D"],
      "correct_answer": "正确答案",
      "knowledge_point_id": "kp-name"
    }}
  ],
  "estimated_mastery_delta": {{
    "kp-name": 预计掌握度提升值(0.0-1.0)
  }}
}}

要求：
1. 题目难度与知识点难度匹配
2. 至少包含 1 道选择题
3. 如果难度 >= 3，可以包含填空题
4. 题面清晰无歧义
5. 选择题提供 4 个选项
