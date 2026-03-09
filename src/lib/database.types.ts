export type Json = string | number | boolean | null | { [key: string]: Json } | Json[]

export interface DiceResult {
  command: string
  result: number[]
  total: number
  text: string
  systemId?: string
}

// Supabase SDK v2が期待するフォーマット
export type Database = {
  public: {
    Tables: {
      rooms: {
        Row: {
          id: string
          name: string
          password: string | null
          gm_name: string | null
          created_at: string
        }
        Insert: {
          id?: string
          name: string
          password?: string | null
          gm_name?: string | null
          created_at?: string
        }
        Update: {
          id?: string
          name?: string
          password?: string | null
          gm_name?: string | null
          created_at?: string
        }
        Relationships: []
      }
      messages: {
        Row: {
          id: string
          room_id: string
          user_name: string
          content: string
          is_dice: boolean
          dice_result: DiceResult | null
          secret: boolean
          created_at: string
        }
        Insert: {
          id?: string
          room_id: string
          user_name: string
          content: string
          is_dice?: boolean
          dice_result?: DiceResult | null
          secret?: boolean
          created_at?: string
        }
        Update: {
          id?: string
          room_id?: string
          user_name?: string
          content?: string
          is_dice?: boolean
          dice_result?: DiceResult | null
          secret?: boolean
          created_at?: string
        }
        Relationships: []
      }
      handouts: {
        Row: {
          id: string
          room_id: string
          title: string
          content: string
          from_user: string
          target_user: string | null
          created_at: string
        }
        Insert: {
          id?: string
          room_id: string
          title: string
          content: string
          from_user: string
          target_user?: string | null
          created_at?: string
        }
        Update: {
          id?: string
          room_id?: string
          title?: string
          content?: string
          from_user?: string
          target_user?: string | null
          created_at?: string
        }
        Relationships: []
      }
      character_sheets: {
        Row: {
          room_id: string
          user_name: string
          data: Record<string, string>
          updated_at: string
        }
        Insert: {
          room_id: string
          user_name: string
          data?: Record<string, string>
          updated_at?: string
        }
        Update: {
          room_id?: string
          user_name?: string
          data?: Record<string, string>
          updated_at?: string
        }
        Relationships: []
      }
      bgm_state: {
        Row: {
          room_id: string
          url: string | null
          is_playing: boolean
          volume: number
          updated_at: string
        }
        Insert: {
          room_id: string
          url?: string | null
          is_playing?: boolean
          volume?: number
          updated_at?: string
        }
        Update: {
          room_id?: string
          url?: string | null
          is_playing?: boolean
          volume?: number
          updated_at?: string
        }
        Relationships: []
      }
      tokens: {
        Row: {
          id: string
          room_id: string
          label: string
          x: number
          y: number
          color: string
          size: number
          created_at: string
        }
        Insert: {
          id?: string
          room_id: string
          label: string
          x?: number
          y?: number
          color?: string
          size?: number
          created_at?: string
        }
        Update: {
          id?: string
          room_id?: string
          label?: string
          x?: number
          y?: number
          color?: string
          size?: number
          created_at?: string
        }
        Relationships: []
      }
    }  // end Tables
    Views: { [_ in never]: never }
    Functions: { [_ in never]: never }
    Enums: { [_ in never]: never }
    CompositeTypes: { [_ in never]: never }
  }
}

export type Room = Database['public']['Tables']['rooms']['Row']
export type Message = Database['public']['Tables']['messages']['Row']
export type Token = Database['public']['Tables']['tokens']['Row']
export type BgmState = Database['public']['Tables']['bgm_state']['Row']
export type CharacterSheet = Database['public']['Tables']['character_sheets']['Row']
export type Handout = Database['public']['Tables']['handouts']['Row']
