'use client'
import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
} from 'react'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface UserProfile {
  user_id: string
  username: string
  avatar_color: string
}

export interface WorkspaceInfo {
  workspace_id: string
  name: string
  description?: string
  role?: string
  member_count?: number
  project_count?: number
  created_at?: number
}

export interface ProjectInfo {
  project_id: string
  workspace_id: string
  name: string
  description?: string
  role?: string
  member_count?: number
  created_at?: number
}

export interface ChatSession {
  session_id: string
  title: string
  mode: string
  workspace_id?: string
  project_id?: string
  messages: any[]
  created_at: number
  updated_at: number
}

export interface AppContextValue {
  // User
  currentUser: UserProfile
  setUsername: (name: string) => void

  // Workspace/Project selection
  selectedWorkspace: WorkspaceInfo | null
  selectedProject: ProjectInfo | null
  selectWorkspace: (ws: WorkspaceInfo | null) => void
  selectProject: (proj: ProjectInfo | null) => void

  // Data lists
  workspaces: WorkspaceInfo[]
  projects: ProjectInfo[]
  loadingWorkspaces: boolean
  loadingProjects: boolean

  // API actions
  refreshWorkspaces: () => Promise<void>
  refreshProjects: (workspace_id: string) => Promise<void>
  createWorkspace: (name: string, description?: string) => Promise<WorkspaceInfo | null>
  createProject: (workspace_id: string, name: string, description?: string) => Promise<ProjectInfo | null>

  // Chat sessions
  sessions: ChatSession[]
  currentSessionId: string | null
  setCurrentSessionId: (id: string | null) => void
  saveSession: (session: ChatSession) => void
  deleteSession: (session_id: string) => void
  getProjectSessions: (workspace_id?: string, project_id?: string) => ChatSession[]
}

// ─── Constants ────────────────────────────────────────────────────────────────

const AVATAR_COLORS = [
  '#7c3aed',
  '#2563eb',
  '#0891b2',
  '#059669',
  '#d97706',
  '#dc2626',
  '#db2777',
  '#7c3aed',
]

const LS_USER = 'ai_agent_user'
const LS_WORKSPACE = 'ai_agent_workspace'
const LS_PROJECT = 'ai_agent_project'
const LS_SESSIONS = 'ai_agent_sessions'
const MAX_SESSIONS = 100

// ─── Helpers ──────────────────────────────────────────────────────────────────

function generateUserId(): string {
  return 'user_' + Math.random().toString(36).slice(2, 10)
}

function pickAvatarColor(user_id: string): string {
  let hash = 0
  for (let i = 0; i < user_id.length; i++) {
    hash = (hash * 31 + user_id.charCodeAt(i)) & 0xffff
  }
  return AVATAR_COLORS[hash % AVATAR_COLORS.length]
}

function createDefaultUser(): UserProfile {
  const user_id = generateUserId()
  return {
    user_id,
    username: '用户' + user_id.slice(-4),
    avatar_color: pickAvatarColor(user_id),
  }
}

function loadFromLS<T>(key: string, fallback: T): T {
  if (typeof window === 'undefined') return fallback
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return fallback
    return JSON.parse(raw) as T
  } catch {
    return fallback
  }
}

function saveToLS(key: string, value: unknown): void {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // quota exceeded – ignore
  }
}

// ─── Context ──────────────────────────────────────────────────────────────────

const AppContext = createContext<AppContextValue | null>(null)

export function AppProvider({ children }: { children: React.ReactNode }) {
  // ── User ──────────────────────────────────────────────────────────────────
  const [currentUser, setCurrentUser] = useState<UserProfile>(() =>
    loadFromLS<UserProfile>(LS_USER, createDefaultUser())
  )

  // Ensure avatar_color is always set (handles old stored users without it)
  useEffect(() => {
    if (!currentUser.avatar_color) {
      const updated = {
        ...currentUser,
        avatar_color: pickAvatarColor(currentUser.user_id),
      }
      setCurrentUser(updated)
      saveToLS(LS_USER, updated)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const setUsername = useCallback((name: string) => {
    setCurrentUser(prev => {
      const updated = { ...prev, username: name.trim() || prev.username }
      saveToLS(LS_USER, updated)
      return updated
    })
  }, [])

  // ── Workspace / Project selection ─────────────────────────────────────────
  const [selectedWorkspace, setSelectedWorkspace] = useState<WorkspaceInfo | null>(() =>
    loadFromLS<WorkspaceInfo | null>(LS_WORKSPACE, null)
  )
  const [selectedProject, setSelectedProject] = useState<ProjectInfo | null>(() =>
    loadFromLS<ProjectInfo | null>(LS_PROJECT, null)
  )

  const selectWorkspace = useCallback((ws: WorkspaceInfo | null) => {
    setSelectedWorkspace(ws)
    saveToLS(LS_WORKSPACE, ws)
    // Clear project when workspace changes
    setSelectedProject(null)
    saveToLS(LS_PROJECT, null)
  }, [])

  const selectProject = useCallback((proj: ProjectInfo | null) => {
    setSelectedProject(proj)
    saveToLS(LS_PROJECT, proj)
  }, [])

  // ── Data lists ────────────────────────────────────────────────────────────
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[]>([])
  const [projects, setProjects] = useState<ProjectInfo[]>([])
  const [loadingWorkspaces, setLoadingWorkspaces] = useState(false)
  const [loadingProjects, setLoadingProjects] = useState(false)

  const refreshWorkspaces = useCallback(async () => {
    setLoadingWorkspaces(true)
    try {
      const res = await fetch(
        `/api/agent/workspaces?user_id=${encodeURIComponent(currentUser.user_id)}`
      )
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json = await res.json()
      setWorkspaces(Array.isArray(json.workspaces) ? json.workspaces : [])
    } catch {
      // silently ignore – keep stale data
    } finally {
      setLoadingWorkspaces(false)
    }
  }, [currentUser.user_id])

  const refreshProjects = useCallback(async (workspace_id: string) => {
    setLoadingProjects(true)
    try {
      const res = await fetch(
        `/api/agent/workspaces/${encodeURIComponent(workspace_id)}/projects`
      )
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json = await res.json()
      setProjects(Array.isArray(json.projects) ? json.projects : [])
    } catch {
      // silently ignore
    } finally {
      setLoadingProjects(false)
    }
  }, [])

  const createWorkspace = useCallback(
    async (name: string, description?: string): Promise<WorkspaceInfo | null> => {
      const res = await fetch('/api/agent/workspaces', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          description: description ?? '',
          creator_id: currentUser.user_id,  // 与服务端 _CreateWorkspaceReq 字段对齐
        }),
      })
      if (!res.ok) {
        const detail = await res.text().catch(() => `HTTP ${res.status}`)
        throw new Error(detail || `HTTP ${res.status}`)
      }
      const json = await res.json()
      const ws: WorkspaceInfo = {
        workspace_id:  json.workspace_id,
        name:          json.name,
        description:   json.description ?? '',
        member_count:  json.member_count ?? 0,
        project_count: json.project_count ?? 0,
      }
      await refreshWorkspaces()
      return ws
    },
    [currentUser.user_id, refreshWorkspaces]
  )

  const createProject = useCallback(
    async (
      workspace_id: string,
      name: string,
      description?: string
    ): Promise<ProjectInfo | null> => {
      const res = await fetch(
        `/api/agent/workspaces/${encodeURIComponent(workspace_id)}/projects`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name,
            description: description ?? '',
            creator_id: currentUser.user_id,  // 与服务端 _CreateProjectReq 字段对齐
          }),
        }
      )
      if (!res.ok) {
        const detail = await res.text().catch(() => `HTTP ${res.status}`)
        throw new Error(detail || `HTTP ${res.status}`)
      }
      const json = await res.json()
      const proj: ProjectInfo = {
        project_id:   json.project_id,
        workspace_id: json.workspace_id ?? workspace_id,
        name:         json.name,
        description:  json.description ?? '',
        member_count: json.member_count ?? 0,
      }
      await refreshProjects(workspace_id)
      return proj
    },
    [currentUser.user_id, refreshProjects]
  )

  // ── Load workspaces on mount ───────────────────────────────────────────────
  const mountedRef = useRef(false)
  useEffect(() => {
    if (mountedRef.current) return
    mountedRef.current = true
    refreshWorkspaces()
  }, [refreshWorkspaces])

  // ── Load projects when selected workspace changes ─────────────────────────
  useEffect(() => {
    if (selectedWorkspace) {
      refreshProjects(selectedWorkspace.workspace_id)
    } else {
      setProjects([])
    }
  }, [selectedWorkspace, refreshProjects])

  // ── Chat sessions ─────────────────────────────────────────────────────────
  const [sessions, setSessions] = useState<ChatSession[]>(() =>
    loadFromLS<ChatSession[]>(LS_SESSIONS, [])
  )
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)

  const saveSession = useCallback((session: ChatSession) => {
    setSessions(prev => {
      const filtered = prev.filter(s => s.session_id !== session.session_id)
      const next = [session, ...filtered]
      const trimmed = next.slice(0, MAX_SESSIONS)
      saveToLS(LS_SESSIONS, trimmed)
      return trimmed
    })
  }, [])

  const deleteSession = useCallback((session_id: string) => {
    setSessions(prev => {
      const next = prev.filter(s => s.session_id !== session_id)
      saveToLS(LS_SESSIONS, next)
      return next
    })
    setCurrentSessionId(prev => (prev === session_id ? null : prev))
  }, [])

  const getProjectSessions = useCallback(
    (workspace_id?: string, project_id?: string): ChatSession[] => {
      if (workspace_id && project_id) {
        return sessions.filter(
          s => s.workspace_id === workspace_id && s.project_id === project_id
        )
      }
      return sessions
    },
    [sessions]
  )

  // ── Context value ─────────────────────────────────────────────────────────
  const value: AppContextValue = {
    currentUser,
    setUsername,
    selectedWorkspace,
    selectedProject,
    selectWorkspace,
    selectProject,
    workspaces,
    projects,
    loadingWorkspaces,
    loadingProjects,
    refreshWorkspaces,
    refreshProjects,
    createWorkspace,
    createProject,
    sessions,
    currentSessionId,
    setCurrentSessionId,
    saveSession,
    deleteSession,
    getProjectSessions,
  }

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp(): AppContextValue {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}
