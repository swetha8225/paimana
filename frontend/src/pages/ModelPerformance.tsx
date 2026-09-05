import { useEffect, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import { api } from '../api'
import { ErrorBox, Loader } from '../components/ui'

function Curves({ m }: { m: any }) {
  if (!m?.curves) return null
  const roc = m.curves.roc.fpr.map((f: number, i: number) => ({ fpr: f, tpr: m.curves.roc.tpr[i] }))
  const pr = m.curves.pr.recall.map((r: number, i: number) => ({ recall: r, precision: m.curves.pr.precision[i] }))
  const cal = m.calibration_curve?.mean_predicted?.map((p: number, i: number) => ({
    predicted: p, empirical: m.calibration_curve.fraction_positive[i],
  })) || []
  return <div className="grid g3">
    <div>
      <h3 style={{ fontSize: 11 }}>ROC (test)</h3>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={roc}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e6ebf2" />
          <XAxis dataKey="fpr" tick={{ fontSize: 9 }} domain={[0, 1]} />
          <YAxis dataKey="tpr" tick={{ fontSize: 9 }} domain={[0, 1]} />
          <Tooltip />
          <Line type="monotone" dataKey="tpr" stroke="#0b2b5b" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
    <div>
      <h3 style={{ fontSize: 11 }}>Precision-Recall (test)</h3>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={pr}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e6ebf2" />
          <XAxis dataKey="recall" tick={{ fontSize: 9 }} domain={[0, 1]} />
          <YAxis dataKey="precision" tick={{ fontSize: 9 }} domain={[0, 1]} />
          <Tooltip />
          <Line type="monotone" dataKey="precision" stroke="#ff9933" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
    <div>
      <h3 style={{ fontSize: 11 }}>Calibration (test)</h3>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={cal}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e6ebf2" />
          <XAxis dataKey="predicted" tick={{ fontSize: 9 }} domain={[0, 1]} />
          <YAxis dataKey="empirical" tick={{ fontSize: 9 }} domain={[0, 1]} />
          <Tooltip />
          <Line type="monotone" dataKey="empirical" stroke="#138808" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  </div>
}

function ClsTable({ results, best }: { results: any; best: string }) {
  const rows = Object.entries(results)
  return <table className="tbl">
    <thead><tr>
      <th>Model</th><th className="num">Accuracy</th><th className="num">Precision</th>
      <th className="num">Recall</th><th className="num">F1</th><th className="num">ROC-AUC</th>
      <th className="num">PR-AUC</th><th className="num">Brier</th>
    </tr></thead>
    <tbody>
      {rows.map(([name, r]: any) => <tr key={name} style={{
        fontWeight: name === best ? 700 : undefined,
        background: name === best ? '#fff7ec' : undefined,
      }}>
        <td>{name}{name === best ? ' ★' : ''}</td>
        {r.error ? <td colSpan={7} style={{ color: 'var(--critical)', fontSize: 11 }}>{r.error}</td> : <>
          <td className="num">{r.accuracy?.toFixed(3)}</td>
          <td className="num">{r.precision?.toFixed(3)}</td>
          <td className="num">{r.recall?.toFixed(3)}</td>
          <td className="num">{r.f1?.toFixed(3)}</td>
          <td className="num">{r.roc_auc?.toFixed(3)}</td>
          <td className="num">{r.pr_auc?.toFixed(3)}</td>
          <td className="num">{r.brier?.toFixed(3)}</td>
        </>}
      </tr>)}
    </tbody>
  </table>
}

export default function ModelPerformance() {
  const [c, setC] = useState<any>(null)
  const [err, setErr] = useState('')
  useEffect(() => { api('/api/model-card').then(setC).catch(e => setErr(e.message)) }, [])
  if (err) return <div className="page"><ErrorBox msg={err} /></div>
  if (!c) return <Loader />
  const t = c.tasks
  const ca = t.classification_approval_stage
  const cm = t.classification_monitoring_stage
  const rc = t.regression_cost_overrun_pct
  const rt = t.regression_time_overrun_months
  const sv = t.survival_time_overrun
  const fc = t.forecast_cost_overrun_pct

  return <div className="page">
    <div className="card mb">
      <h3>Validation policy</h3>
      <div className="note" style={{ fontSize: 12.5 }}>{c.split_policy.description}</div>
      <ul style={{ fontSize: 12, color: 'var(--muted)', margin: '8px 0 0', paddingLeft: 18 }}>
        {c.split_policy.leakage_notes.map((l: string, i: number) => <li key={i} style={{ marginBottom: 4 }}>{l}</li>)}
      </ul>
    </div>

    <div className="card mb">
      <h3>Statistical vs Machine Learning — computed on the actual dataset</h3>
      <table className="tbl">
        <thead><tr><th>Task</th><th>Statistical baseline</th><th>ML winner</th><th>Winner (computed)</th></tr></thead>
        <tbody>
          {Object.entries(c.stat_vs_ml).map(([k, v]: any) => <tr key={k}>
            <td>{k.replace(/_/g, ' ')}</td>
            <td>{v?.statistical ? `${v.statistical.model}: ${v.statistical.roc_auc?.toFixed(3) ?? v.statistical.mae?.toFixed(2) ?? '—'}` : '—'}</td>
            <td>{v?.ml ? `${v.ml.model}: ${v.ml.roc_auc?.toFixed(3) ?? v.ml.mae?.toFixed(2) ?? '—'}` : '—'}</td>
            <td style={{ fontWeight: 700, color: v?.winner_on_test_roc_auc || v?.winner_on_test_mae ? 'var(--navy)' : undefined }}>
              {v?.winner_on_test_roc_auc || v?.winner_on_test_mae || '—'}</td>
          </tr>)}
        </tbody>
      </table>
      <div className="note" style={{ marginTop: 6 }}>Winners are computed from held-out test metrics on identical splits — never hard-coded.</div>
    </div>

    <div className="card mb">
      <h3>Cost-overrun classification — approval stage</h3>
      {ca?.available ? <>
        <div className="note mb">
          Train months: {ca.train_months.join(', ')} · Validation: {ca.val_months.join(', ')} · Test: {ca.test_months.join(', ')}
          (n = {ca.n_train.toLocaleString()} / {ca.n_val.toLocaleString()} / {ca.n_test.toLocaleString()}; positive rate {(100 * ca.positive_rate).toFixed(1)}%).
          Flag = current cost exceeds original (MoSPI counting). Best model: <b>{ca.best_model}</b>, probability-calibrated ({ca.best_metrics.calibration_method}) on the validation split.
        </div>
        <ClsTable results={ca.model_results} best={ca.best_model} />
        <div className="mt"><Curves m={ca.best_metrics} /></div>
        {ca.project_disjoint?.available && <div className="card mt" style={{ background: '#f0f6ff' }}>
          <h3>Strict project-disjoint validation</h3>
          <div className="note mb">{ca.project_disjoint.description}</div>
          <table className="tbl">
            <thead><tr><th>Model</th><th className="num">ROC-AUC</th><th className="num">Accuracy</th>
              <th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th></tr></thead>
            <tbody><tr>
              <td>{ca.project_disjoint.model}</td>
              <td className="num" style={{ fontWeight: 700 }}>{ca.project_disjoint.roc_auc.toFixed(3)}</td>
              <td className="num">{ca.project_disjoint.accuracy.toFixed(3)}</td>
              <td className="num">{ca.project_disjoint.precision.toFixed(3)}</td>
              <td className="num">{ca.project_disjoint.recall.toFixed(3)}</td>
              <td className="num">{ca.project_disjoint.f1.toFixed(3)}</td>
            </tr></tbody>
          </table>
          <div className="note" style={{ marginTop: 6 }}>This is the honest figure for screening a newly approved project the model has never seen.</div>
        </div>}
      </> : <div className="insufficient">{ca?.reason}</div>}
    </div>

    <div className="card mb">
      <h3>Cost-overrun classification — monitoring stage</h3>
      {cm?.available ? <>
        <div className="note mb">Adds monitoring information available at the report month (project age, elapsed schedule fraction, expenditure pattern). Best: <b>{cm.best_model}</b>.</div>
        <ClsTable results={cm.model_results} best={cm.best_model} />
      </> : <div className="insufficient">{cm?.reason}</div>}
    </div>

    <div className="grid g2 mb">
      <div className="card">
        <h3>Cost-overrun magnitude (regression)</h3>
        {rc?.available ? <>
          <table className="tbl">
            <thead><tr><th>Model</th><th className="num">MAE (pp)</th><th className="num">RMSE</th><th className="num">R²</th></tr></thead>
            <tbody>
              {Object.entries(rc.model_results).map(([n, r]: any) => <tr key={n} style={{
                fontWeight: n === rc.best_model ? 700 : undefined, background: n === rc.best_model ? '#fff7ec' : undefined,
              }}>
                {r.error ? <td colSpan={4} style={{ fontSize: 11 }}>{r.error}</td> : <>
                  <td>{n}</td><td className="num">{r.mae?.toFixed(2)}</td>
                  <td className="num">{r.rmse?.toFixed(2)}</td><td className="num">{r.r2?.toFixed(3)}</td>
                </>}
              </tr>)}
            </tbody>
          </table>
          <div className="note mt">Target mean {rc.target_mean?.toFixed(1)}pp ± {rc.target_std?.toFixed(1)}. Test on {rc.test_months.join(', ')}.</div>
        </> : <div className="insufficient">{rc?.reason}</div>}
      </div>
      <div className="card">
        <h3>Schedule delay (regression on reported delay)</h3>
        {rt?.available ? <>
          <table className="tbl">
            <thead><tr><th>Model</th><th className="num">MAE (months)</th><th className="num">RMSE</th><th className="num">R²</th></tr></thead>
            <tbody>
              {Object.entries(rt.model_results).map(([n, r]: any) => <tr key={n} style={{
                fontWeight: n === rt.best_model ? 700 : undefined, background: n === rt.best_model ? '#fff7ec' : undefined,
              }}>
                {r.error ? <td colSpan={4} style={{ fontSize: 11 }}>{r.error}</td> : <>
                  <td>{n}</td><td className="num">{r.mae?.toFixed(2)}</td>
                  <td className="num">{r.rmse?.toFixed(2)}</td><td className="num">{r.r2?.toFixed(3)}</td>
                </>}
              </tr>)}
            </tbody>
          </table>
          <div className="note mt">Target: anticipated delay vs original target, as reported by MoSPI.</div>
        </> : <div className="insufficient">{rt?.reason}</div>}
      </div>
    </div>

    <div className="card mb">
      <h3>Time-overrun approach — auto-decision</h3>
      {sv?.available ? <div className="note" style={{ fontSize: 12.5 }}>
        <b>Survival analysis selected.</b> {sv.decision}
        <table className="tbl mt">
          <thead><tr><th>Model</th><th className="num">CV concordance</th></tr></thead>
          <tbody>{Object.entries(sv.model_results).map(([n, r]: any) => <tr key={n}>
            <td>{n}</td><td className="num">{r.concordance_index_cv_mean?.toFixed(3)} ± {r.concordance_index_cv_std?.toFixed(3)}</td>
          </tr>)}</tbody>
        </table>
      </div> : <div className="insufficient">
        {sv?.reason}
        <div style={{ marginTop: 6 }}>Fallback in use: regression on reported schedule delay (table above).</div>
      </div>}
    </div>

    <div className="card mb">
      <h3>Forecasting — honest evaluation</h3>
      {fc?.available ? <>
        <div className="note mb" style={{ fontSize: 12.5 }}>{fc.feasibility?.note}</div>
        <table className="tbl">
          <thead><tr><th>Hold-out month</th><th className="num">n</th><th className="num">Panel model MAE</th>
            <th className="num">Naive baseline MAE</th><th>Panel model beats baseline?</th></tr></thead>
          <tbody><tr>
            <td>{fc.test_month}</td><td className="num">{fc.n_test?.toLocaleString()}</td>
            <td className="num">{fc.model_mae?.toFixed(2)}</td>
            <td className="num">{fc.naive_mae?.toFixed(2)}</td>
            <td style={{ fontWeight: 700, color: fc.model_beats_naive ? 'var(--low)' : 'var(--critical)' }}>
              {fc.model_beats_naive ? 'Yes' : 'No — naive/observed trend is used'}</td>
          </tr></tbody>
        </table>
        <div className="note mt">{fc.note}</div>
      </> : <div className="insufficient">{fc?.reason}</div>}
    </div>

    {c.shap_global && <div className="card mb">
      <h3>What influences predictions most (global SHAP)</h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={c.shap_global.features.map((f: string, i: number) => ({
          feature: f, importance: c.shap_global.mean_abs_shap[i],
        }))} layout="vertical" margin={{ left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e6ebf2" />
          <XAxis type="number" tick={{ fontSize: 11 }} />
          <YAxis type="category" dataKey="feature" width={160} tick={{ fontSize: 11 }} />
          <Tooltip />
          <Bar dataKey="importance" fill="#0b2b5b" radius={[0, 3, 3, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <div className="note">{c.shap_global.caveat}</div>
    </div>}

    <div className="card">
      <h3>Variables the current public data does NOT contain</h3>
      <div className="note mb">PAIMANA never fabricates features. These are documented as recommended future collections (proposed OCMS/PAIMANA data extensions):</div>
      <ul style={{ fontSize: 12.5, margin: 0, paddingLeft: 18, color: 'var(--text)' }}>
        {c.future_variables.map((v: string, i: number) => <li key={i} style={{ marginBottom: 4 }}>{v}</li>)}
      </ul>
    </div>
  </div>
}
