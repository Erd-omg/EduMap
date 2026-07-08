-- Sample data for EduMap development
INSERT INTO users (external_id, display_name) VALUES
    ('dev-user-001', '张三'),
    ('dev-user-002', '李四')
ON CONFLICT (external_id) DO NOTHING;
