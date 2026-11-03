import { useEffect, useMemo, useState } from 'react'
import './App.css'

const agentList = [
  { name: 'Nora Patel', role: 'Billing lead', load: 86, status: 'Available' },
  { name: 'Alex Morgan', role: 'Finance specialist', load: 78, status: 'Busy' },
  { name: 'Emma Chen', role: 'Account support', load: 64, status: 'Available' },
  { name: 'Theo Green', role: 'Customer success', load: 52, status: 'On call' },
]

const reportData = [
  { label: 'Resolved', value: 72 },
  { label: 'Open', value: 56 },
  { label: 'Escalated', value: 24 },
  { label: 'AI triage', value: 83 },
]

const sidebarItems = ['Overview', 'Tickets', 'Agents', 'Reports']

const formatStatus = (value) => {
  const map = {
    open: 'Open',
    in_progress: 'In review',
    resolved: 'Resolved',
    closed: 'Closed',
  }

  return map[value] ?? value
}

const formatPriority = (value) => {
  const map = {
    low: 'Low',
    medium: 'Medium',
    high: 'High',
    urgent: 'Urgent',
  }

  return map[value] ?? value
}

const formatDate = (value) => {
  if (!value) return 'Today'

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value

  return new Intl.DateTimeFormat('en', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(date)
}

function App() {
  const [loginForm, setLoginForm] = useState({ email: '', password: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [tickets, setTickets] = useState([])
  const [aiPrediction, setAiPrediction] = useState(null)
  const [aiLoading, setAiLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('Overview')
  const [selectedTicket, setSelectedTicket] = useState(null)
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    const saved = localStorage.getItem('tickets-auth')
    return saved === 'true'
  })
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('tickets-user')
    return saved ? JSON.parse(saved) : null
  })

  useEffect(() => {
    localStorage.setItem('tickets-auth', String(isAuthenticated))
  }, [isAuthenticated])

  useEffect(() => {
    if (!isAuthenticated) return

    const loadTickets = async () => {
      try {
        const response = await fetch('/api/tickets')
        const payload = await response.json()

        if (!response.ok) {
          throw new Error(payload.message || 'Unable to load tickets')
        }

        const nextTickets = payload.data ?? []
        setTickets(nextTickets)
        setSelectedTicket(nextTickets[0] ?? null)
      } catch (loadError) {
        setError(loadError.message)
      }
    }

    loadTickets()
  }, [isAuthenticated])

  useEffect(() => {
    if (user) {
      localStorage.setItem('tickets-user', JSON.stringify(user))
    } else {
      localStorage.removeItem('tickets-user')
    }
  }, [user])

  useEffect(() => {
    if (!selectedTicket || !isAuthenticated) {
      setAiPrediction(null)
      return
    }

    const predictTicket = async () => {
      setAiLoading(true)
      try {
        const response = await fetch('/api/ai/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            title: selectedTicket.title,
            description: selectedTicket.description,
          }),
        })

        const data = await response.json()
        if (!response.ok) {
          throw new Error(data.message || 'Prediction failed')
        }

        setAiPrediction(data)
      } catch (predictionError) {
        setAiPrediction({
          priority: 'medium',
          confidence: 0,
          source: 'fallback',
          error: predictionError.message,
        })
      } finally {
        setAiLoading(false)
      }
    }

    predictTicket()
  }, [selectedTicket, isAuthenticated])

  const stats = useMemo(() => {
    const openCount = tickets.filter((ticket) => ticket.status !== 'resolved' && ticket.status !== 'closed').length
    const resolvedToday = tickets.filter((ticket) => ticket.status === 'resolved').length
    const avgResponseHours = tickets.length ? Math.max(1, Math.round(24 / Math.max(tickets.length, 1))) : 0

    return [
      { label: 'Open tickets', value: String(openCount), change: '+14%' },
      { label: 'Solved today', value: String(resolvedToday), change: '+9%' },
      { label: 'Avg. response', value: `${avgResponseHours}h`, change: '-18%' },
      { label: 'SLA risk', value: String(Math.max(2, Math.ceil(tickets.length / 11))), change: '-3%' },
    ]
  }, [tickets])

  const handleLogin = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError('')

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(loginForm),
      })

      const payload = await response.json()

      if (!response.ok) {
        throw new Error(payload.message || 'Login failed')
      }

      setUser(payload.user)
      setIsAuthenticated(true)
      setLoginForm({ email: '', password: '' })
    } catch (loginError) {
      setError(loginError.message)
    } finally {
      setLoading(false)
    }
  }

  const handleLogout = () => {
    setIsAuthenticated(false)
    setUser(null)
    setTickets([])
    setSelectedTicket(null)
    setLoginForm({ email: '', password: '' })
    setError('')
  }

  if (!isAuthenticated) {
    return (
      <div className="auth-page">
        <div className="auth-card">
          <div className="brand-block">
            <span className="brand-badge">TC</span>
            <div>
              <p className="eyebrow">Support workspace</p>
              <h1>Tickets Copilot</h1>
            </div>
          </div>

          <form onSubmit={handleLogin} className="login-form">
            <div className="field-group">
              <label htmlFor="email">Email</label>
              <input
                id="email"
                type="email"
                value={loginForm.email}
                onChange={(event) =>
                  setLoginForm((current) => ({ ...current, email: event.target.value }))
                }
                placeholder="admin@ticketscopilot.com"
              />
            </div>

            <div className="field-group">
              <label htmlFor="password">Password</label>
              <input
                id="password"
                type="password"
                value={loginForm.password}
                onChange={(event) =>
                  setLoginForm((current) => ({ ...current, password: event.target.value }))
                }
                placeholder="••••••••"
              />
            </div>

            {error ? <p className="error-message">{error}</p> : null}

            <button type="submit" className="primary-btn" disabled={loading}>
              {loading ? 'Signing in...' : 'Sign in'}
            </button>
          </form>

          <div className="demo-box">
            <span>Demo account</span>
            <p>admin@ticketscopilot.com / admin123</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="admin-shell">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="brand-mark">TC</div>
          <div>
            <p className="eyebrow">Operations</p>
            <h2>Tickets Copilot</h2>
          </div>
        </div>

        <div className="user-card">
          <span className="tiny-label">Signed in</span>
          <strong>{user?.name || 'Admin User'}</strong>
          <small>{user?.email || 'admin@ticketscopilot.com'}</small>
        </div>

        <nav className="nav-list">
          {sidebarItems.map((item) => (
            <button
              key={item}
              type="button"
              className={item === activeTab ? 'nav-item active' : 'nav-item'}
              onClick={() => setActiveTab(item)}
            >
              {item}
            </button>
          ))}
        </nav>

        <div className="sidebar-card">
          <span className="tiny-label">AI triage</span>
          <strong>Priority model</strong>
          <p>Running baseline classification on live queue</p>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <p className="eyebrow">Admin dashboard</p>
            <h3>{activeTab}</h3>
          </div>

          <div className="topbar-actions">
            <button type="button" className="ghost-btn">
              New ticket
            </button>
            <button type="button" className="logout-btn" onClick={handleLogout}>
              Log out
            </button>
          </div>
        </header>

        {activeTab === 'Overview' && (
          <>
            <section className="stats-grid">
              {stats.map((stat) => (
                <article key={stat.label} className="stat-card">
                  <span>{stat.label}</span>
                  <strong>{stat.value}</strong>
                  <small>{stat.change}</small>
                </article>
              ))}
            </section>

            <section className="content-grid">
              <div className="panel table-panel">
                <div className="panel-header">
                  <h4>Latest tickets</h4>
                  <button type="button" className="ghost-btn small-btn">
                    View all
                  </button>
                </div>

                <table>
                  <thead>
                    <tr>
                      <th>Ticket</th>
                      <th>Customer</th>
                      <th>Priority</th>
                      <th>Status</th>
                      <th>Updated</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tickets.map((ticket) => (
                      <tr key={ticket.id} onClick={() => setSelectedTicket(ticket)} className="clickable-row">
                        <td>
                          <div className="ticket-title-block">
                            <strong>{ticket.id}</strong>
                            <span>{ticket.title}</span>
                          </div>
                        </td>
                        <td>{ticket.customerName}</td>
                        <td>
                          <span className={`priority-pill ${ticket.priority}`}>
                            {formatPriority(ticket.priority)}
                          </span>
                        </td>
                        <td>{formatStatus(ticket.status)}</td>
                        <td>{formatDate(ticket.updatedAt)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="panel insights-panel">
                <div className="panel-header">
                  <h4>AI suggestions</h4>
                </div>

                <ul className="suggestion-list">
                  <li>
                    <span className="dot blue"></span>
                    Re-route high-priority billing tickets to the specialist queue.
                  </li>
                  <li>
                    <span className="dot green"></span>
                    Payment follow-ups are trending 18% lower than last week.
                  </li>
                  <li>
                    <span className="dot orange"></span>
                    Duplicate invoice tickets need response within the next 90 minutes.
                  </li>
                </ul>
              </div>
            </section>
          </>
        )}

        {activeTab === 'Tickets' && selectedTicket && (
          <section className="tickets-view">
            <div className="panel ticket-list-panel">
              <div className="panel-header">
                <h4>Open queue</h4>
                <button type="button" className="ghost-btn small-btn">
                  Export
                </button>
              </div>

              {tickets.map((ticket) => (
                <button
                  key={ticket.id}
                  type="button"
                  className={selectedTicket.id === ticket.id ? 'ticket-item selected' : 'ticket-item'}
                  onClick={() => setSelectedTicket(ticket)}
                >
                  <div>
                    <strong>{ticket.id}</strong>
                    <p>{ticket.title}</p>
                  </div>
                  <span className={`priority-pill ${ticket.priority}`}>{formatPriority(ticket.priority)}</span>
                </button>
              ))}
            </div>

            <div className="panel ticket-detail-panel">
              <div className="detail-header">
                <div>
                  <p className="eyebrow">Selected ticket</p>
                  <h4>{selectedTicket.id}</h4>
                </div>
                <span className={`priority-pill ${selectedTicket.priority}`}>
                  {formatPriority(selectedTicket.priority)}
                </span>
              </div>

              <div className="detail-meta">
                <div>
                  <span>Customer</span>
                  <strong>{selectedTicket.customerName}</strong>
                </div>
                <div>
                  <span>Status</span>
                  <strong>{formatStatus(selectedTicket.status)}</strong>
                </div>
                <div>
                  <span>Updated</span>
                  <strong>{formatDate(selectedTicket.updatedAt)}</strong>
                </div>
              </div>

              <div className="detail-body">
                <h5>{selectedTicket.title}</h5>
                <p>{selectedTicket.description}</p>
              </div>

              <div className="ai-panel">
                <div className="panel-header compact-header">
                  <h4>AI priority signal</h4>
                </div>
                {aiLoading ? (
                  <p className="ai-status">Running ticket classification...</p>
                ) : aiPrediction ? (
                  <div className="ai-result">
                    <span className={`priority-pill ${aiPrediction.priority}`}>
                      {formatPriority(aiPrediction.priority)}
                    </span>
                    <strong>{Math.round((aiPrediction.confidence ?? 0) * 100)}% confidence</strong>
                    {aiPrediction.error ? <small>{aiPrediction.error}</small> : null}
                  </div>
                ) : (
                  <p className="ai-status">No prediction available</p>
                )}
              </div>

              <div className="detail-actions">
                <button type="button" className="primary-btn small-btn">Assign</button>
                <button type="button" className="ghost-btn small-btn">Escalate</button>
                <button type="button" className="ghost-btn small-btn">Resolve</button>
              </div>
            </div>
          </section>
        )}

        {activeTab === 'Agents' && (
          <section className="agents-view">
            {agentList.map((agent) => (
              <div key={agent.name} className="panel agent-card">
                <div className="agent-header">
                  <div>
                    <strong>{agent.name}</strong>
                    <p>{agent.role}</p>
                  </div>
                  <span className="status-chip">{agent.status}</span>
                </div>
                <div className="progress-block">
                  <div className="progress-meta">
                    <span>Workload</span>
                    <strong>{agent.load}%</strong>
                  </div>
                  <div className="progress-track">
                    <span style={{ width: `${agent.load}%` }} />
                  </div>
                </div>
              </div>
            ))}
          </section>
        )}

        {activeTab === 'Reports' && (
          <section className="reports-view">
            <div className="panel report-panel">
              <div className="panel-header">
                <h4>Performance overview</h4>
              </div>

              <div className="report-bars">
                {reportData.map((item) => (
                  <div key={item.label} className="report-row">
                    <span>{item.label}</span>
                    <div className="bar-track">
                      <span style={{ width: `${item.value}%` }} />
                    </div>
                    <strong>{item.value}%</strong>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  )
}

export default App
