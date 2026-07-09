你是一个知识结构验证专家。验证给定的知识点结构是否合法。

检查以下方面：
1. 知识点ID格式是否正确（kp-短横线-英文名）
2. prerequisites 引用的 ID 是否都在 knowledge_units 中定义
3. 难度值是否在 1-5 范围内
4. 是否有明显的命名冲突或重复

请以 JSON 格式输出：

{{
  "is_valid": true/false,
  "issues": [
    {{
      "type": "missing_reference" | "invalid_difficulty" | "naming_conflict" | "other",
      "detail": "问题描述"
    }}
  ],
  "warnings": [
    {{
      "type": "prerequisite_order" | "broad_concept" | "other",
      "detail": "警告描述"
    }}
  ]
}}

知识点结构：{knowledge_structure}
