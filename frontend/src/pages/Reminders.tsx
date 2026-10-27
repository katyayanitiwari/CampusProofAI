import { useEffect, useMemo, useState } from 'react';
import { Bell, CalendarDays, Check, Clock3, ExternalLink, LoaderCircle, Plus, Trash2 } from 'lucide-react';
import { call } from '../services/api';

type Reminder = {
  id: number;
  title: string;
  category: string;
  due_date: string;
  notes: string;
  source_url: string;
  completed: number;
};

const categories = [
  ['deadline', 'Deadline'],
  ['application', 'Application'],
  ['follow-up', 'Follow-up'],
  ['document', 'Document'],
  ['other', 'Other'],
];

function localToday() {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
    .format(new Date(`${value}T00:00:00`));
}

function dateLabel(value: string) {
  const diff = Math.round((new Date(`${value}T00:00:00`).getTime() - new Date(`${localToday()}T00:00:00`).getTime()) / 86400000);
  if (diff < 0) return 'Past due';
  if (diff === 0) return 'Today';
  if (diff === 1) return 'Tomorrow';
  if (diff <= 7) return `In ${diff} days`;
  return '';
}

export default function Reminders() {
  const [items, setItems] = useState<Reminder[]>([]);
  const [filter, setFilter] = useState<'active' | 'completed' | 'all'>('active');
  const [title, setTitle] = useState('');
  const [category, setCategory] = useState('deadline');
  const [dueDate, setDueDate] = useState('');
  const [notes, setNotes] = useState('');
  const [sourceUrl, setSourceUrl] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  async function refresh() {
    const rows = await call('/reminders');
    setItems(rows as Reminder[]);
  }

  useEffect(() => {
    refresh().catch((e) => setError((e as Error).message)).finally(() => setLoading(false));
  }, []);

  const activeItems = useMemo(() => items.filter((item) => !item.completed), [items]);
  const nextItem = activeItems[0];
  const visibleItems = items.filter((item) => filter === 'all' || (filter === 'completed' ? !!item.completed : !item.completed));

  async function addReminder(event: React.FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError('');
    try {
      await call('/reminders', {
        title: title.trim(),
        category,
        due_date: dueDate,
        notes: notes.trim(),
        source_url: sourceUrl.trim(),
      });
      setTitle('');
      setCategory('deadline');
      setDueDate('');
      setNotes('');
      setSourceUrl('');
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function toggleReminder(item: Reminder) {
    setError('');
    try {
      await call(`/reminders/${item.id}`, { completed: !item.completed }, 'PATCH');
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function removeReminder(item: Reminder) {
    setError('');
    try {
      await call(`/reminders/${item.id}`, undefined, 'DELETE');
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return <div className="reminders-page">
    <section className="reminders-intro">
      <div>
        <span className="overline">YOUR OPPORTUNITY CALENDAR</span>
        <h2>Keep the dates that matter in view.</h2>
        <p>Save application deadlines, follow-ups, and documents to check. Your list stays with your CampusProof account on this laptop.</p>
      </div>
      <div className="reminder-illustration" aria-hidden="true">
        <div className="calendar-page"><div className="calendar-rings"><i/><i/></div><span>JUN</span><b>24</b><small>application<br/>deadline</small></div>
        <span className="calendar-spark spark-a">✳</span><span className="calendar-spark spark-b">·</span>
      </div>
    </section>

    <div className="reminder-summary">
      <div className="summary-label"><Bell size={16}/><span>ACTIVE REMINDERS</span></div>
      <b>{activeItems.length}</b>
      <p>{activeItems.length === 1 ? 'One item to keep an eye on.' : 'Items still on your list.'}</p>
      <div className="summary-next"><span>NEXT DATE</span><strong>{nextItem ? formatDate(nextItem.due_date) : 'Nothing scheduled'}</strong>{nextItem&&<small>{nextItem.title}</small>}</div>
    </div>

    <div className="reminder-layout">
      <section className="panel reminder-form-panel">
        <div className="reminder-form-heading"><div className="reminder-plus"><Plus size={17}/></div><div><h3>Add a reminder</h3><p>Keep the original source nearby for later.</p></div></div>
        <form onSubmit={addReminder}>
          <label>What should you remember?<input required minLength={2} maxLength={120} value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Example: Submit the scholarship form"/></label>
          <div className="reminder-fields-row">
            <label>Type<select value={category} onChange={(e) => setCategory(e.target.value)}>{categories.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label>
            <label>When<input required type="date" min={localToday()} value={dueDate} onChange={(e) => setDueDate(e.target.value)}/></label>
          </div>
          <label>Notes <span className="optional-label">OPTIONAL</span><textarea rows={3} maxLength={1000} value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Eligibility detail, contact name, or what you need to prepare"/></label>
          <label>Reference link <span className="optional-label">OPTIONAL</span><input type="url" maxLength={500} value={sourceUrl} onChange={(e) => setSourceUrl(e.target.value)} placeholder="https://official-site.example"/></label>
          {error&&<div className="reminder-error">{error}</div>}
          <button className="primary reminder-save" disabled={saving||!title.trim()||!dueDate}>{saving?<LoaderCircle className="spin" size={16}/>:<Plus size={16}/>} Save reminder</button>
          <small className="reminder-note">Reminders appear here when you open the portal. They do not send phone or email notifications.</small>
        </form>
      </section>

      <section className="panel reminder-list-panel">
        <div className="reminder-list-head"><div><span className="overline">LOOK BACK OR PLAN AHEAD</span><h3>Your reminders</h3></div><CalendarDays size={19}/></div>
        <div className="reminder-filters">{(['active', 'completed', 'all'] as const).map((value) => <button key={value} className={filter===value?'selected':''} onClick={() => setFilter(value)}>{value==='active'?'Active':value==='completed'?'Completed':'All'}{value==='active'&&activeItems.length>0&&<span>{activeItems.length}</span>}</button>)}</div>
        {loading?<div className="reminder-empty"><LoaderCircle className="spin" size={19}/> Loading your list…</div>:visibleItems.length===0?<div className="reminder-empty"><Clock3 size={22}/><b>{filter==='completed'?'Nothing completed yet.':'No reminders on this list.'}</b><span>Add a date on the left to keep an opportunity in view.</span></div>:<div className="reminder-items">{visibleItems.map((item) => {
          const label = item.completed ? 'Completed' : dateLabel(item.due_date);
          const categoryName = categories.find(([value]) => value===item.category)?.[1]||'Reminder';
          return <article className={`reminder-item ${item.completed?'is-complete':''}`} key={item.id}>
            <button className="reminder-check" aria-label={item.completed?'Mark active':'Mark complete'} onClick={() => toggleReminder(item)}>{item.completed?<Check size={15}/>:null}</button>
            <div className="reminder-item-body"><div className="reminder-item-meta"><span className="reminder-category">{categoryName}</span><span className={`reminder-date ${label==='Past due'?'is-overdue':label==='Today'?'is-today':''}`}>{label&&<Clock3 size={12}/>} {label||formatDate(item.due_date)}</span></div><h4>{item.title}</h4><div className="reminder-due">{formatDate(item.due_date)}</div>{item.notes&&<p>{item.notes}</p>}{item.source_url&&<a className="reminder-source" href={item.source_url} target="_blank" rel="noreferrer">Open saved reference <ExternalLink size={13}/></a>}</div>
            <button className="reminder-delete" aria-label="Remove reminder" onClick={() => removeReminder(item)}><Trash2 size={15}/></button>
          </article>;
        })}</div>}
      </section>
    </div>
  </div>;
}

