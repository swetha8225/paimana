import { useEffect, useState } from 'react'
import { api, monthLabel } from '../api'
import { ErrorBox, Loader } from '../components/ui'

export default function DataQuality() {
  const [d, setD] = useState<any>(null)
  const [err, setErr] = useState('')
  useEffect(() => { api('/api/data-quality').then(setD).catch(e => setErr(e.message)) }, [])
  if (err) return <div className="page"><ErrorBox msg={err} /></div>
  if (!d) return <Loader />

  const m = d.manifest || {}
  // the manifest uses singular "validation" and "notes"
  const v = m.validation || m.validations || {}
  const cor = v.computed_vs_reported_cost_overrun || {}
  const tor = v.computed_vs_reported_time_overrun || {}
  const cons = v.original_cost_consistent_across_months || {}
  const notes: string[] = m.notes || []

  return <div className="page">
    <div className="card mb">
      <h3>Dataset overview</h3>
      <div className="grid g4">
        <div className="kpi">
          <div className="label">Unique projects</div>
          <div className="value">{(v.unique_projects ?? d.manifest?.validation?.unique_projects ?? 0).toLocaleString()}</div>
          <div className="sub">central-sector, ₹150 Cr+</div>
        </div>
        <div className="kpi">
          <div className="label">Project-month records</div>
          <div className="value">{(v.rows ?? 0).toLocaleString()}</div>
          <div className="sub">across {v.report_months?.length ?? 0} report months</div>
        </div>
        <div className="kpi">
          <div className="label">Source reports</div>
          <div className="value">{Object.keys(m.sources || {}).length}</div>
          <div className="sub">all public MoSPI PDFs</div>
        </div>
        <div className="kpi">
          <div className="label">Report months</div>
          <div className="value" style={{ fontSize: 15, lineHeight: 1.5 }}>
            {(v.report_months || []).map((mm: string) => monthLabel(mm)).join(" · ")}
          </div>
        </div>
      </div>
    </div>

    <div className="grid g2 mb">
      <div className="card">
        <h3>Source reports (public MoSPI data)</h3>
        <div className="tbl-wrap">
          <table className="tbl">
            <thead><tr><th>Report month</th><th>Report &amp; PDF link</th></tr></thead>
            <tbody>
              {Object.entries(m.sources || {}).map(([month, s]: any) => <tr key={month}>
                <td style={{ whiteSpace: 'nowrap' }}>{monthLabel(month)}</td>
                <td>
                  <a href={s.url} target="_blank" rel="noreferrer" style={{ color: 'var(--navy)' }}>{s.report}</a>
                  <div style={{ fontSize: 10.5, color: 'var(--muted)', wordBreak: 'break-all' }}>{s.url}</div>
                </td>
              </tr>)}
            </tbody>
          </table>
        </div>
        <div className="note mt">Every number in PAIMANA is parsed from these official publications — nothing is simulated.</div>
      </div>

      <div className="card">
        <h3>Parse validation — computed vs printed in the PDFs</h3>
        <table className="tbl">
          <tbody>
            <tr>
              <td>Cost-overrun % matched within 1pp</td>
              <td className="num" style={{ fontWeight: 700, color: 'var(--low)' }}>
                {cor.match_within_1pp_pct != null ? `${cor.match_within_1pp_pct}%` : '—'}
                <span style={{ fontWeight: 400, color: 'var(--muted)' }}>
                  {' '}(n = {(cor.n ?? 0).toLocaleString()})
                </span>
              </td>
            </tr>
            <tr>
              <td>Time-overrun matched within 1 month</td>
              <td className="num" style={{ fontWeight: 700, color: 'var(--low)' }}>
                {tor.match_within_1m_pct != null ? `${tor.match_within_1m_pct}%` : '—'}
                <span style={{ fontWeight: 400, color: 'var(--muted)' }}>
                  {' '}(n = {(tor.n ?? 0).toLocaleString()})
                </span>
              </td>
            </tr>
            <tr>
              <td>Original cost consistent across months</td>
              <td className="num" style={{ fontWeight: 700 }}>
                {cons.consistent_pct != null ? `${cons.consistent_pct}%` :
                  (d.original_cost_consistency_pct != null ? `${d.original_cost_consistency_pct.toFixed(1)}%` : '—')}
                <span style={{ fontWeight: 400, color: 'var(--muted)' }}>
                  {' '}({(cons.projects ?? 0).toLocaleString()} projects)
                </span>
              </td>
            </tr>
          </tbody>
        </table>
        <div className="note mt">Values are recomputed from the parsed panel and compared against the overruns printed
          in the source tables. Residual inconsistencies reflect genuine restatements in MoSPI reports.</div>

        <h3 style={{ marginTop: 14 }}>Field coverage (share missing)</h3>
        <div className="tbl-wrap">
          <table className="tbl">
            <tbody>
              {Object.entries(d.null_share_pct || {}).map(([k, val]: any) => <tr key={k}>
                <td>{k.replace(/_/g, ' ')}</td>
                <td className="num" style={{
                  color: val > 30 ? 'var(--high)' : val > 10 ? 'var(--moderate)' : 'var(--low)',
                  fontWeight: 600,
                }}>{val.toFixed(1)}%</td>
              </tr>)}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <div className="card mb">
      <h3>Records per report month</h3>
      <div className="tbl-wrap">
        <table className="tbl">
          <thead><tr>
            <th>Month</th><th className="num">Ongoing census</th><th className="num">Completed</th>
            <th className="num">Closed unfinished</th><th className="num">Total</th>
          </tr></thead>
          <tbody>
            {d.per_month?.map((r: any) => <tr key={r.report_month}>
              <td style={{ whiteSpace: 'nowrap' }}>{monthLabel(r.report_month)}</td>
              <td className="num">{(r.ongoing_report || 0).toLocaleString()}</td>
              <td className="num">{(r.completed || 0).toLocaleString()}</td>
              <td className="num">{(r.closed_unfinished || 0).toLocaleString()}</td>
              <td className="num" style={{ fontWeight: 700 }}>
                {((r.ongoing_report || 0) + (r.completed || 0) + (r.closed_unfinished || 0)).toLocaleString()}</td>
            </tr>)}
          </tbody>
        </table>
      </div>
    </div>

    <div className="grid g2">
      <div className="card">
        <h3>Methodology notes &amp; formula definitions</h3>
        <ul style={{ fontSize: 12.5, margin: 0, paddingLeft: 18 }}>
          {notes.map((n, i) => <li key={i} style={{ marginBottom: 6 }}>{n}</li>)}
        </ul>
      </div>
      <div className="card">
        <h3>Known limitations (documented honestly)</h3>
        <ul style={{ fontSize: 12.5, margin: 0, paddingLeft: 18 }}>
          {d.known_limitations?.map((l: string, i: number) => <li key={i} style={{ marginBottom: 6 }}>{l}</li>)}
        </ul>
      </div>
    </div>
  </div>
}
