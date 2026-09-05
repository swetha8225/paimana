import { useEffect, useState } from 'react'
import { api } from '../api'
import { ErrorBox, Loader } from '../components/ui'

export default function RiskMethodology() {
  const [m, setM] = useState<any>(null)
  const [w, setW] = useState<any>(null)
  const [msg, setMsg] = useState('')
  const [err, setErr] = useState('')
  useEffect(() => { api('/api/risk/methodology').then(setM).catch(e => setErr(e.message)) }, [])
  if (err) return <div className="page"><ErrorBox msg={err} /></div>
  if (!m) return <Loader />

  const setWeight = (k: string, v: string) =>
    setW({ ...(w || m.weights), [k]: parseFloat(v) || 0 })

  const save = async () => {
    setMsg('')
    try {
      const r = await api('/api/config/risk-weights', {
        method: 'PUT', body: JSON.stringify(w),
      })
      setMsg(`Weights updated — project levels recomputed. Sum = ${Object.values(r.weights).reduce((s: number, x: any) => s + x, 0).toFixed(2)}`)
      setW(r.weights)
    } catch (e: any) { setMsg(`⚠ ${e.message}`) }
  }

  const current = w || m.weights

  return <div className="page" style={{ maxWidth: 900 }}>
    <div className="card mb">
      <h3>Purpose</h3>
      <div className="note" style={{ fontSize: 12.5 }}>
        The PAIMANA implementation-risk score summarises, for each project at its
        most recent report month, how much trouble the project is in and how much
        trouble the model expects. It is deliberately transparent: every component
        is a formula over observable quantities and model outputs — no hidden
        model drives the score itself.
      </div>
      <div className="insufficient" style={{ marginTop: 10 }}>{m.disclaimer}</div>
    </div>

    <div className="card mb">
      <h3>Component formulas</h3>
      <table className="tbl">
        <thead><tr><th>Component</th><th>Computation</th></tr></thead>
        <tbody>
          {Object.entries(m.components).map(([k, v]: any) => <tr key={k}>
            <td style={{ fontWeight: 600 }}>{k.replace(/_/g, ' ')}</td>
            <td style={{ fontSize: 12 }}>{v}</td>
          </tr>)}
        </tbody>
      </table>
    </div>

    <div className="card mb">
      <h3>Weights (configurable)</h3>
      <div className="note mb">Total score = Σ weight × component. Levels: Low &lt; 30, Moderate 30–55, High 55–75, Critical ≥ 75.</div>
      <div className="whatif-grid">
        {Object.keys(m.components).map(k => <div key={k}>
          <label>{k.replace(/_/g, ' ')} (current {current[k]})</label>
          <input type="number" step="0.05" min="0" max="1" value={current[k]}
            onChange={e => setWeight(k, e.target.value)} />
        </div>)}
      </div>
      <button onClick={save} style={{
        marginTop: 10, padding: '8px 18px', background: 'var(--navy)', color: '#fff',
        border: 'none', borderRadius: 8, cursor: 'pointer', fontWeight: 600,
      }}>Save & recompute project levels</button>
      <button onClick={() => setW(m.weights)} style={{
        marginTop: 10, marginLeft: 8, padding: '8px 18px', background: '#fff',
        border: '1px solid var(--border)', borderRadius: 8, cursor: 'pointer',
      }}>Reset form</button>
      {msg && <div className="note" style={{ marginTop: 8, color: 'var(--navy)' }}>{msg}</div>}
      <div className="note" style={{ marginTop: 8 }}>
        Default weights: cost 0.35 / schedule 0.35 / expenditure 0.20 / reporting 0.10.
        Weights must sum to ~1.0.
      </div>
    </div>

    <div className="card">
      <h3>How the score relates to model predictions</h3>
      <div className="note" style={{ fontSize: 12.5 }}>
        The score blends <b>observed</b> indicators (reported overrun %, months past
        target, expenditure patterns, data gaps) with the <b>model's calibrated
        probability</b> of cost overrun. The prediction models themselves never
        use post-outcome information as inputs — see the leakage policy on the
        Model Performance page. SHAP explanations describe influence on model
        predictions, not causal effects.
      </div>
    </div>
  </div>
}
