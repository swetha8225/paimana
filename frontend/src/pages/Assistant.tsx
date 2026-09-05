import { useEffect, useRef, useState } from 'react'
import { api } from '../api'

export default function Assistant() {
  const [msgs, setMsgs] = useState<{ role: 'user' | 'bot'; text: string }[]>([{
    role: 'bot',
    text: 'Namaste! I am the PAIMANA assistant. Ask me about MoSPI central-sector projects — risk rankings, sector/ministry/state aggregates, cost & schedule overruns, model drivers, data quality, or specific projects.\n\nExamples: "top 10 riskiest projects", "cost overrun in Railways", "tell me about project N24001451".',
  }])
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  const ask = async (text?: string) => {
    const question = (text ?? q).trim()
    if (!question || busy) return
    setMsgs(m => [...m, { role: 'user', text: question }])
    setQ(''); setBusy(true)
    try {
      const r = await api(`/api/assistant?q=${encodeURIComponent(question)}`)
      setMsgs(m => [...m, { role: 'bot', text: r.answer }])
    } catch (e: any) {
      setMsgs(m => [...m, { role: 'bot', text: `Sorry, an error occurred: ${e.message}` }])
    }
    setBusy(false)
  }

  const suggestions = ['Top 10 riskiest projects', 'Cost overrun in Railways sector',
    'Projects in Maharashtra', 'What drives the model', 'Data quality and sources']

  return <div className="page" style={{ maxWidth: 860 }}>
    <div className="card chat" style={{ minHeight: 420 }}>
      {msgs.map((m, i) => <div key={i} className={`msg ${m.role}`}>{m.text}</div>)}
      {busy && <div className="msg bot">Thinking…</div>}
      <div ref={endRef} />
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 4 }}>
        {suggestions.map(s => <button key={s} className="tag-sim" style={{
          background: '#fff', cursor: 'pointer', fontSize: 11.5,
        }} onClick={() => ask(s)}>{s}</button>)}
      </div>
      <div className="chat-input">
        <input value={q} onChange={e => setQ(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && ask()}
          placeholder="Ask about projects, sectors, overruns, risk…" />
        <button onClick={() => ask()} disabled={busy}>Ask</button>
      </div>
      <div className="note" style={{ marginTop: 8 }}>
        Answers are computed live from the MoSPI panel in the PAIMANA database via
        SQL queries. The assistant does not state or imply official government
        decisions; where data is insufficient it says so explicitly.
      </div>
    </div>
  </div>
}
