'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import { Card, CardContent, Badge } from '@/components/ui';
import { clearColdStartCache } from '@/hooks/use-cold-start';
import { getUserId } from '@/lib/user-id';

/* ── Types ───────────────────────────────────────────────────────── */

interface ResourceItem {
  id: string;
  name: string;
  type: string;
  source: 'user_upload' | 'system_generated';
  kp_id?: string | null;
  kp_name?: string | null;
  file_size?: number | null;
  description?: string | null;
  created_at: string;
  parse_status?: string | null;   // "pending" | "parsing" | "parsed" | "error"
  parse_stats?: { chunks: number; chars: number; indexed: number; error?: string } | null;
}

interface ChunkPreview {
  index: number;
  text_preview: string;
  char_count: number;
}

type ResourceFilter = 'all' | 'upload' | 'explanation' | 'exercise' | 'visualization' | 'code';

/* ── Constants ───────────────────────────────────────────────────── */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

const TYPE_LABELS: Record<string, string> = {
  all: '全部',
  upload: '上传文档',
  explanation: '讲解',
  exercise: '练习',
  visualization: '可视化',
  code: '代码',
};

const SOURCE_LABELS: Record<string, string> = {
  user_upload: '用户上传',
  system_generated: '系统生成',
};

const TYPE_COLORS: Record<string, string> = {
  upload: 'bg-blue-100 text-blue-700',
  explanation: 'bg-green-100 text-green-700',
  exercise: 'bg-purple-100 text-purple-700',
  visualization: 'bg-orange-100 text-orange-700',
  code: 'bg-rose-100 text-rose-700',
};

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

/* ── Resource Library Component ───────────────────────────────────── */

interface ResourceLibraryProps {
  refreshKey?: number;
  /** Optional KP to bind newly uploaded files to (from the /generate KP picker). */
  defaultKpId?: string | null;
  defaultKpName?: string | null;
}

export function ResourceLibrary({ refreshKey = 0, defaultKpId, defaultKpName }: ResourceLibraryProps) {
  const [resources, setResources] = useState<ResourceItem[]>([]);
  const [filter, setFilter] = useState<ResourceFilter>('all');
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [feedback, setFeedback] = useState<{ type: 'success' | 'error'; msg: string } | null>(null);
  const [previewResource, setPreviewResource] = useState<ResourceItem | null>(null);
  const [previewChunks, setPreviewChunks] = useState<ChunkPreview[]>([]);
  const [previewLoading, setPreviewLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const showFeedback = useCallback((type: 'success' | 'error', msg: string) => {
    setFeedback({ type, msg });
    setTimeout(() => setFeedback(null), 5000);
  }, []);

  // ── Fetch resources ──────────────────────────────────────
  const fetchResources = useCallback(async () => {
    try {
      const params = new URLSearchParams({ user_id: getUserId(), limit: '200' });
      if (filter !== 'all') params.set('type', filter);
      const res = await fetch(`${API_BASE}/api/v1/resources?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResources(data.resources || []);
    } catch (err) {
      console.error('Failed to fetch resources:', err);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    fetchResources();
  }, [fetchResources, refreshKey]);

  // ── Upload handler ───────────────────────────────────────
  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files?.length) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', files[0]);
    formData.append('user_id', getUserId());
    // Bind the upload to the currently selected KP so it shows in the graph.
    if (defaultKpId) formData.append('kp_id', defaultKpId);
    if (defaultKpName) formData.append('kp_name', defaultKpName);

    try {
      const res = await fetch(`${API_BASE}/api/v1/resources/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        showFeedback('error', `上传失败: ${err.detail || res.statusText}`);
      } else {
        showFeedback('success', '上传成功');
        clearColdStartCache();
        fetchResources();
      }
    } catch (err) {
      showFeedback('error', `上传出错: ${err instanceof Error ? err.message : '未知错误'}`);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }, [fetchResources, defaultKpId, defaultKpName]);

  // ── Preview content fetch handler ────────────────────────
  // Shows the persisted description (system-generated resources) plus any
  // parsed chunks (uploaded files).
  const handlePreview = useCallback(async (resource: ResourceItem) => {
    setPreviewResource(resource);
    setPreviewChunks([]);
    setPreviewLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/resources/${resource.id}/chunks?limit=10`);
      if (res.ok) {
        const data = await res.json();
        setPreviewChunks(data.chunks || []);
      }
    } catch {
      // Ignore fetch errors
    } finally {
      setPreviewLoading(false);
    }
  }, []);

  // ── Delete handler ───────────────────────────────────────
  const handleDelete = useCallback(async (id: string) => {
    if (!confirm('确定删除该资源？')) return;
    try {
      const res = await fetch(`${API_BASE}/api/v1/resources/${id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setResources((prev) => prev.filter((r) => r.id !== id));
    } catch (err) {
      showFeedback('error', `删除失败: ${err instanceof Error ? err.message : '未知错误'}`);
    }
  }, []);

  // ── Filtered resources ───────────────────────────────────
  const filtered = filter === 'all' ? resources : resources.filter((r) => r.type === filter);
  const isEmpty = !loading && filtered.length === 0;

  return (
    <div className="space-y-6">
      {/* Upload area */}
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-semibold text-text-primary">上传学习资料</h3>
              <p className="mt-1 text-xs text-text-secondary">
                支持 PDF、Markdown、Word、PPT、TXT、代码文件等格式
              </p>
            </div>
            <div className="relative">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.md,.docx,.pptx,.txt,.py,.js,.ts,.html,.csv"
                onChange={handleUpload}
                className="hidden"
                id="file-upload"
                disabled={uploading}
              />
              <label
                htmlFor="file-upload"
                className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-brand px-5 py-2.5 text-sm font-medium text-white hover:bg-brand-hover disabled:bg-text-light transition-colors"
              >
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                </svg>
                {uploading ? '上传中...' : '选择文件上传'}
              </label>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Filters */}
      <div className="flex items-center gap-2 flex-wrap">
        {(Object.entries(TYPE_LABELS) as [ResourceFilter, string][]).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setFilter(key)}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
              filter === key
                ? 'bg-brand text-white'
                : 'bg-bg-secondary text-text-secondary hover:bg-border'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Resource list */}
      <div className="space-y-2">
        {loading ? (
          <div className="flex items-center justify-center py-12 text-sm text-text-light">
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-brand border-t-transparent mr-2" />
            加载中...
          </div>
        ) : isEmpty ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <div className="mb-3 text-4xl">📂</div>
              <p className="text-sm text-text-secondary">暂无资源</p>
              <p className="mt-1 text-xs text-text-light">
                {filter === 'all' ? '点击上方按钮上传学习资料' : `没有${TYPE_LABELS[filter]}类型的资源`}
              </p>
            </CardContent>
          </Card>
        ) : (
          filtered.map((resource) => (
            <Card key={resource.id}>
              <CardContent
                className="flex items-center gap-4 p-4 cursor-pointer hover:bg-bg-secondary/50 transition-colors"
                onClick={() => handlePreview(resource)}
              >
                {/* Type icon */}
                <div
                  className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-sm ${
                    TYPE_COLORS[resource.type] || 'bg-bg-secondary text-text-secondary'
                  }`}
                >
                  {resource.type === 'upload' ? '📄' : resource.type === 'explanation' ? '📝' : resource.type === 'exercise' ? '✏️' : resource.type === 'visualization' ? '📊' : '💻'}
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-text-primary truncate">
                      {resource.name}
                    </span>
                    <Badge variant="default" size="sm">
                      {TYPE_LABELS[resource.type] || resource.type}
                    </Badge>
                    <span className="text-[10px] text-text-light">
                      {SOURCE_LABELS[resource.source] || resource.source}
                    </span>
                  </div>
                  <div className="mt-0.5 flex items-center gap-3 text-xs text-text-light">
                    {resource.file_size != null && <span>{formatFileSize(resource.file_size)}</span>}
                    {(resource.kp_name || resource.kp_id) && <span>知识点: {resource.kp_name || resource.kp_id}</span>}
                    <span>{formatDate(resource.created_at)}</span>
                    {resource.source === 'user_upload' && resource.parse_status && (
                      <span className={`inline-flex items-center gap-1 ${
                        resource.parse_status === 'parsed' ? 'text-success' :
                        resource.parse_status === 'error' ? 'text-danger' :
                        resource.parse_status === 'parsing' ? 'text-brand' :
                        'text-text-light'
                      }`}>
                        {resource.parse_status === 'parsing' && '⏳ 解析中...'}
                        {resource.parse_status === 'parsed' && `✅ 已解析 (${resource.parse_stats?.chunks || 0} 段)`}
                        {resource.parse_status === 'error' && '⚠️ 解析失败'}
                        {resource.parse_status === 'pending' && '⏸ 等待解析'}
                      </span>
                    )}
                  </div>
                </div>

                {/* Actions */}
                <button
                  onClick={(e) => { e.stopPropagation(); handlePreview(resource); }}
                  className="rounded-lg p-2 text-text-light hover:bg-brand/10 hover:text-brand transition-colors"
                  title="查看内容"
                >
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); handleDelete(resource.id); }}
                  className="rounded-lg p-2 text-text-light hover:bg-danger-bg hover:text-danger transition-colors"
                  title="删除"
                >
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                  </svg>
                </button>
              </CardContent>
            </Card>
          ))
        )}
      </div>

      {/* Chunk Preview Modal */}
      {previewResource && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
          onClick={() => { setPreviewResource(null); setPreviewChunks([]); }}
        >
          <div
            className="max-h-[80vh] w-full max-w-2xl overflow-y-auto rounded-xl bg-white p-6 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-semibold text-text-primary">
                资源内容 — {previewResource.name}
              </h3>
              <button
                onClick={() => { setPreviewResource(null); setPreviewChunks([]); }}
                className="rounded-lg p-1.5 text-text-light hover:bg-bg-secondary hover:text-text-primary transition-colors"
              >
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {previewLoading ? (
              <div className="flex items-center justify-center py-8 text-sm text-text-light">
                <div className="h-5 w-5 animate-spin rounded-full border-2 border-brand border-t-transparent mr-2" />
                加载内容...
              </div>
            ) : (
              <div className="space-y-4">
                {/* Generated resources carry their markdown body as description */}
                {previewResource.description && (
                  <div>
                    <p className="text-xs font-medium text-text-secondary mb-1.5">内容</p>
                    <div className="rounded-lg border border-border bg-bg-secondary p-3 text-xs text-text-primary whitespace-pre-wrap leading-relaxed max-h-72 overflow-y-auto">
                      {previewResource.description}
                    </div>
                  </div>
                )}
                {/* Uploaded files expose parsed chunks */}
                {previewChunks.length > 0 ? (
                  <div>
                    <p className="text-xs text-text-secondary mb-1.5">
                      共 {previewChunks.length} 个文本段落（每段预览前 200 字符）
                    </p>
                    <div className="space-y-3">
                      {previewChunks.map((chunk, i) => (
                        <div key={i} className="rounded-lg border border-border bg-bg-secondary p-3">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-medium text-text-secondary">
                              段落 #{i + 1}
                            </span>
                            <span className="text-[10px] text-text-light">
                              {chunk.char_count} 字符
                            </span>
                          </div>
                          <p className="text-xs text-text-primary whitespace-pre-wrap leading-relaxed">
                            {chunk.text_preview}
                            {chunk.char_count > 200 && '...'}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : !previewResource.description ? (
                  <div className="text-center py-8">
                    <p className="text-sm text-text-light mb-1">
                      {previewResource.parse_stats?.chunks && previewResource.parse_stats?.indexed === 0
                        ? `文件已解析 ${previewResource.parse_stats.chunks} 个段落，但向量数据库未连接，无法预览具体内容`
                        : '暂无内容'}
                    </p>
                  </div>
                ) : null}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
