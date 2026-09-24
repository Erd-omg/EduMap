-- EduMap memory system tables
-- Episodic + Semantic memory + Anti-gaming persistence
-- See src/memory/models.py for corresponding Pydantic models.

-- Episodic memory: individual interaction records
CREATE TABLE IF NOT EXISTS episodic_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(50) NOT NULL,  -- 'mentor_query','quiz_answer','generation','review','profile_update'
    input TEXT,
    output TEXT,
    metadata JSONB DEFAULT '{}',
    importance_score FLOAT DEFAULT 0.5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Embedding of the interaction text, used by LongTermMemory.recall_relevant
-- to rank history by semantic relevance to the current query (the
-- recency+importance+relevance formula in src/memory/recall_scoring.py).
-- Stored as a JSONB array of floats rather than pgvector: the embedding is
-- produced by the in-process sentence-transformers model, and scoring happens
-- in Python over a small candidate window, so a vector index would add an
-- extension dependency without buying anything at this scale.
-- NULL for rows written before this column existed — recall_relevant treats a
-- missing embedding as relevance 0, so those rows can still be recalled.
-- `IF NOT EXISTS` keeps this file idempotent for databases created earlier.
ALTER TABLE episodic_memory
    ADD COLUMN IF NOT EXISTS embedding JSONB;

CREATE INDEX IF NOT EXISTS idx_episodic_user_created
    ON episodic_memory(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_user_type
    ON episodic_memory(user_id, event_type);
CREATE INDEX IF NOT EXISTS idx_episodic_session
    ON episodic_memory(session_id);

-- Semantic memory: distilled knowledge about users and domain
CREATE TABLE IF NOT EXISTS semantic_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    memory_type VARCHAR(50) NOT NULL,  -- 'user_preference','domain_knowledge','skill_summary','learning_style'
    key VARCHAR(255) NOT NULL,
    value JSONB NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, memory_type, key)
);
CREATE INDEX IF NOT EXISTS idx_semantic_user_type
    ON semantic_memory(user_id, memory_type);

-- Anti-gaming state (replaces in-memory dict in anti_gaming.py)
CREATE TABLE IF NOT EXISTS anti_gaming_state (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    kp_id VARCHAR(255) NOT NULL DEFAULT '',
    alpha DOUBLE PRECISION DEFAULT 2.0,
    beta DOUBLE PRECISION DEFAULT 2.0,
    last_event_time TIMESTAMP WITH TIME ZONE,
    pattern_flags JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, kp_id)
);

-- Evaluation results for RAG quality tracking
CREATE TABLE IF NOT EXISTS evaluation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    eval_type VARCHAR(50) NOT NULL,  -- 'retrieval','generation','end_to_end'
    metrics JSONB NOT NULL,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_eval_user_type
    ON evaluation_results(user_id, eval_type);

-- Learning progress (replaces in-memory _progress dict in path_service.py)
CREATE TABLE IF NOT EXISTS learning_progress (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    kp_id VARCHAR(255) NOT NULL,
    status VARCHAR(20) DEFAULT 'not_started',  -- 'not_started','in_progress','completed','reviewing'
    score FLOAT,
    time_spent_minutes INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, kp_id)
);
CREATE INDEX IF NOT EXISTS idx_progress_user_status
    ON learning_progress(user_id, status);

-- Tool execution log
CREATE TABLE IF NOT EXISTS tool_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255),
    tool_name VARCHAR(100) NOT NULL,
    input JSONB NOT NULL,
    output JSONB,
    success BOOLEAN DEFAULT true,
    duration_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tool_session
    ON tool_executions(session_id);

-- Triggers
CREATE TRIGGER trg_semantic_memory_updated_at
    BEFORE UPDATE ON semantic_memory
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_anti_gaming_state_updated_at
    BEFORE UPDATE ON anti_gaming_state
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_learning_progress_updated_at
    BEFORE UPDATE ON learning_progress
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
