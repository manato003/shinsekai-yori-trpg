/**
 * BCDice API クライアント
 * Docs: https://bcdice.trpg.net (v2 API)
 *
 * フォールバック: BCDice APIが使えない場合は汎用ダイスパーサーを使う
 */

const BCDICE_API = process.env.NEXT_PUBLIC_BCDICE_API_URL ?? 'https://bcdice.trpg.net'

interface BCDiceRollResponse {
  ok: boolean
  text?: string
  secret?: boolean
  success?: boolean
  failure?: boolean
  critical?: boolean
  fumble?: boolean
  rands?: Array<{ faces: number; value: number }>
}

export interface DiceRollResult {
  command: string
  text: string
  secret: boolean
  success?: boolean
  failure?: boolean
  rands: Array<{ faces: number; value: number }>
  total: number
}

/**
 * BCDice APIでダイスコマンドを実行する
 * @param command - "2d6" "1d100" "CoC:1d100" etc.
 * @param systemId - BCDiceゲームシステムID (default: "DiceBot")
 */
export async function rollDice(
  command: string,
  systemId = 'DiceBot'
): Promise<DiceRollResult | null> {
  try {
    const params = new URLSearchParams({ command })
    const res = await fetch(`${BCDICE_API}/v2/game_system/${systemId}/roll?${params}`)

    if (!res.ok) {
      // BCDice不認識コマンド → フォールバック
      return fallbackRoll(command)
    }

    const data: BCDiceRollResponse = await res.json()
    if (!data.ok || !data.text) return null

    const rands = data.rands ?? []
    const total = rands.reduce((acc, r) => acc + r.value, 0)

    return {
      command,
      text: data.text,
      secret: data.secret ?? false,
      success: data.success,
      failure: data.failure,
      rands,
      total,
    }
  } catch {
    // ネットワークエラー → フォールバック
    return fallbackRoll(command)
  }
}

/**
 * 汎用ダイスパーサー (BCDice不使用時のフォールバック)
 * 対応: NdM, NdM+K, NdM-K
 */
function fallbackRoll(command: string): DiceRollResult | null {
  const match = command.trim().match(/^(\d+)[dD](\d+)([+-]\d+)?$/i)
  if (!match) return null

  const count = parseInt(match[1], 10)
  const faces = parseInt(match[2], 10)
  const modifier = match[3] ? parseInt(match[3], 10) : 0

  if (count < 1 || count > 100 || faces < 1) return null

  const rands = Array.from({ length: count }, () => ({
    faces,
    value: Math.floor(Math.random() * faces) + 1,
  }))

  const sum = rands.reduce((acc, r) => acc + r.value, 0)
  const total = sum + modifier

  const rollStr = rands.map((r) => r.value).join('+')
  const modStr = modifier !== 0 ? (modifier > 0 ? `+${modifier}` : `${modifier}`) : ''
  const text = `${command.toUpperCase()} → ${rollStr}${modStr} = ${total}`

  return { command, text, secret: false, rands, total }
}

/**
 * チャット入力がダイスコマンドかどうか判定する
 * BCDiceのコマンド形式: 英数字始まりで d を含む
 */
export function isDiceCommand(input: string): boolean {
  const trimmed = input.trim()
  // xDy, xdy, SystemName:command などの形式
  return /^[a-zA-Z0-9]*\d+[dD]\d+/i.test(trimmed) || /^[A-Za-z]+:/i.test(trimmed)
}

/** BCDice対応ゲームシステム一覧を取得 */
export async function fetchGameSystems(): Promise<Array<{ id: string; name: string; sortKey: string }>> {
  const res = await fetch(`${BCDICE_API}/v2/game_system`)
  if (!res.ok) return []
  const data = await res.json()
  // APIはsnake_case (sort_key) で返すのでcamelCaseにマップする
  return (data.game_system ?? []).map((s: { id: string; name: string; sort_key: string }) => ({
    id: s.id,
    name: s.name,
    sortKey: s.sort_key ?? '',
  }))
}
