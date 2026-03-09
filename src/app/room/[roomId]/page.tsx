import { SessionRoom } from '@/components/session/SessionRoom'

interface Props {
  params: Promise<{ roomId: string }>
}

export default async function RoomPage({ params }: Props) {
  const { roomId } = await params
  return <SessionRoom roomId={roomId} />
}
