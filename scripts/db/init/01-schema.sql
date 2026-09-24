-- EduMap PostgreSQL initialization
-- Creates base tables for the profile service

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User profiles table
-- user_id is VARCHAR to support external/user-supplied IDs (e.g. "anonymous")
-- without requiring a FK to the users table. For production, add a proper
-- auth system and restore the FK.
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    profile_data JSONB NOT NULL DEFAULT '{}',
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Learning behavior log
CREATE TABLE IF NOT EXISTS learning_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    resource_id VARCHAR(255),
    action_type VARCHAR(50) NOT NULL,
    metadata JSONB DEFAULT '{}',
    duration_seconds INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Forgetting-curve review log — append-only history of every scored review.
--
-- Why this exists: ``forgetting_curve_state`` keeps only the *latest* fitted
-- parameters per (user, kp), and ``learning_progress`` is an upsert that
-- overwrites.  Neither retains the (time, score) series that any spaced-
-- repetition model needs in order to be fitted or evaluated — which is why
-- the forgetting curve could not be validated against real data.  This table
-- is that missing series.
--
-- Notes on the shape:
--   * ``user_id`` is VARCHAR with no FK, matching ``forgetting_curve_state``
--     and ``episodic_memory``.  Anonymous/demo users do not exist in ``users``
--     and an FK here would reject their reviews.
--   * Append-only: rows are never updated or deleted (except by the privacy
--     deletion path).  A review that happened stays in the history.
--   * ``score`` is normalised to [0, 1] — it is fed straight into the
--     forgetting-curve update, so storing raw per-quiz maxima would require
--     the reader to re-normalise and risk inconsistency.
CREATE TABLE IF NOT EXISTS forgetting_review_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    kp_id VARCHAR(255) NOT NULL,
    score FLOAT NOT NULL,             -- normalised [0, 1]
    -- 'learn' | 'review' | 'quiz' — distinguishes first exposure from a
    -- later review, which spaced-repetition models weight differently.
    event_type VARCHAR(32) NOT NULL DEFAULT 'review',
    -- 'assessment' | 'path' | 'mentor' — which subsystem produced the review.
    source VARCHAR(32) NOT NULL DEFAULT 'assessment',
    reviewed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Learning path snapshots
CREATE TABLE IF NOT EXISTS learning_paths (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    path_data JSONB NOT NULL,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_learning_logs_user_id ON learning_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_learning_logs_created_at ON learning_logs(created_at);
-- The evaluation/fitting query is "one user's reviews for one kp, in time
-- order", so the composite index is ordered to serve that scan directly.
CREATE INDEX IF NOT EXISTS idx_review_log_user_kp_time
    ON forgetting_review_log(user_id, kp_id, reviewed_at);
CREATE INDEX IF NOT EXISTS idx_review_log_reviewed_at
    ON forgetting_review_log(reviewed_at);
CREATE INDEX IF NOT EXISTS idx_learning_paths_user_id ON learning_paths(user_id);
CREATE INDEX IF NOT EXISTS idx_learning_paths_active ON learning_paths(active);
CREATE INDEX IF NOT EXISTS idx_learning_paths_user_active ON learning_paths(user_id, active);

-- Path recommendations (每次推荐结果落库，支撑通知与历史回溯)
CREATE TABLE IF NOT EXISTS path_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    course_id VARCHAR(255) NOT NULL,
    kp_id VARCHAR(255) NOT NULL,
    kp_name VARCHAR(512),
    reason TEXT,
    recommended_content_type VARCHAR(50),
    estimated_session_min INTEGER,
    source VARCHAR(50) NOT NULL DEFAULT 'path',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_path_reco_user_id ON path_recommendations(user_id);
CREATE INDEX IF NOT EXISTS idx_path_reco_user_created ON path_recommendations(user_id, created_at DESC);

-- Forgetting curve state (Ebbinghaus + Bayesian parameters)
CREATE TABLE IF NOT EXISTS forgetting_curve_state (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    kp_id VARCHAR(255) NOT NULL,
    alpha DOUBLE PRECISION NOT NULL DEFAULT 2.0,
    beta DOUBLE PRECISION NOT NULL DEFAULT 2.0,
    last_review_time TIMESTAMP WITH TIME ZONE,
    review_count INTEGER NOT NULL DEFAULT 0,
    strength DOUBLE PRECISION NOT NULL DEFAULT 24.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, kp_id)
);
CREATE INDEX IF NOT EXISTS idx_forgetting_user_id ON forgetting_curve_state(user_id);

-- Resource metadata table (replaces in-memory _resources dict)
CREATE TABLE IF NOT EXISTS resources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL DEFAULT 'anonymous',
    name VARCHAR(512) NOT NULL,
    type VARCHAR(50) NOT NULL DEFAULT 'upload',
    source VARCHAR(50) NOT NULL DEFAULT 'user_upload',
    kp_id VARCHAR(255),
    kp_name VARCHAR(512),
    file_size INTEGER,
    file_path TEXT,
    description TEXT,
    parse_status VARCHAR(20) DEFAULT 'pending',
    parse_stats JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_resources_user_id ON resources(user_id);
CREATE INDEX IF NOT EXISTS idx_resources_kp_id ON resources(kp_id);
CREATE INDEX IF NOT EXISTS idx_resources_type ON resources(type);

-- Auto-update updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_user_profiles_updated_at
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
