'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { supabase } from '@/lib/supabase'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader } from '@/components/ui/card'

export default function HomePage() {
  const router = useRouter()
  const [tab, setTab] = useState<'create' | 'join'>('create')

  const [roomName, setRoomName] = useState('')
  const [password, setPassword] = useState('')
  const [roomId, setRoomId] = useState('')
  const [joinPassword, setJoinPassword] = useState('')
  const [userName, setUserName] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault()
    if (!roomName.trim() || !userName.trim()) return
    setLoading(true)
    setError('')
    try {
      const { data, error: err } = await supabase
        .from('rooms')
        .insert({ name: roomName.trim(), password: password.trim() || null, gm_name: userName.trim() })
        .select('id')
        .single()
      if (err) throw err
      sessionStorage.setItem('trpg_user_name', userName.trim())
      router.push(`/room/${data.id}`)
    } catch (err) {
      setError('ルームの作成に失敗しました。Supabaseの設定を確認してください。')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  async function handleJoin(e: React.FormEvent) {
    e.preventDefault()
    if (!roomId.trim() || !userName.trim()) return
    setLoading(true)
    setError('')
    try {
      const { data, error: err } = await supabase
        .from('rooms')
        .select('id, password')
        .eq('id', roomId.trim())
        .single()
      if (err || !data) { setError('ルームが見つかりません。'); return }
      if (data.password && data.password !== joinPassword) {
        setError('パスワードが間違っています。'); return
      }
      sessionStorage.setItem('trpg_user_name', userName.trim())
      router.push(`/room/${data.id}`)
    } catch (err) {
      setError('接続エラーが発生しました。')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4">
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold tracking-tight mb-2">
          <span className="text-primary">⚔</span> TRPGセッション
        </h1>
        <p className="text-muted-foreground text-sm">
          どのシステムでも使えるオンラインセッションツール
        </p>
      </div>

      <Card className="w-full max-w-md">
        <CardHeader className="pb-3">
          <div className="flex rounded-md overflow-hidden border border-border">
            {(['create', 'join'] as const).map((t) => (
              <button
                key={t}
                onClick={() => { setTab(t); setError('') }}
                className={`flex-1 py-2 text-sm font-medium transition-colors ${
                  tab === t
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {t === 'create' ? 'ルームを作成' : 'ルームに参加'}
              </button>
            ))}
          </div>
        </CardHeader>

        <CardContent>
          <form onSubmit={tab === 'create' ? handleCreate : handleJoin} className="space-y-3">
            <Field label="あなたの名前" value={userName} onChange={setUserName} placeholder="キャラクター名 / ハンドル" required />

            {tab === 'create' ? (
              <>
                <Field label="ルーム名" value={roomName} onChange={setRoomName} placeholder="例: SW2.5 冒険の夜" required />
                <Field label="パスワード（任意）" value={password} onChange={setPassword} placeholder="なしでも可" type="password" />
              </>
            ) : (
              <>
                <Field label="ルームID" value={roomId} onChange={setRoomId} placeholder="GMから共有されたUUID" required />
                <Field label="パスワード" value={joinPassword} onChange={setJoinPassword} placeholder="設定されている場合" type="password" />
              </>
            )}

            {error && <p className="text-sm text-destructive">{error}</p>}

            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? '処理中...' : tab === 'create' ? 'ルームを作成して入室' : '参加する'}
            </Button>
          </form>
        </CardContent>
      </Card>

      <p className="text-xs text-muted-foreground mt-6">
        BCDice対応 · チャット · ダイス · リアルタイム同期
      </p>
    </div>
  )
}

function Field({
  label, value, onChange, placeholder, type = 'text', required,
}: {
  label: string; value: string; onChange: (v: string) => void
  placeholder?: string; type?: string; required?: boolean
}) {
  return (
    <div>
      <label className="text-xs text-muted-foreground mb-1 block">{label}</label>
      <Input
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        required={required}
      />
    </div>
  )
}
