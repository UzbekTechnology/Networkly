import socket, threading, struct, zlib, uuid, math, time
import ctypes
import ctypes.wintypes
import win32gui
import win32con
import win32api
from enum import IntEnum

screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

active_entities = {}
player_position = (0.0, 0.0, 0.0)
player_camera = (0.0, 0.0)
base_fov = 90
player_fov = base_fov + 30

state_lock = threading.Lock()
entities_lock = threading.Lock()


def wnd_proc(hwnd, msg, wparam, lparam):
    if msg == win32con.WM_TIMER:
        draw_entities(hwnd)
        return 0
    elif msg == win32con.WM_DESTROY:
        ctypes.windll.user32.KillTimer(hwnd, 1)
        win32gui.PostQuitMessage(0)
        return 0
    else:
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

def create_fullscreen_window():
    class_name = "Window"
    wnd_class = win32gui.WNDCLASS()
    wnd_class.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
    wnd_class.lpfnWndProc = wnd_proc
    wnd_class.hInstance = win32api.GetModuleHandle()
    wnd_class.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
    wnd_class.hbrBackground = win32con.COLOR_WINDOW
    wnd_class.lpszClassName = class_name
    win32gui.RegisterClass(wnd_class)

    hwnd = win32gui.CreateWindowEx(
        win32con.WS_EX_LAYERED | win32con.WS_EX_TOPMOST | win32con.WS_EX_TRANSPARENT,
        class_name, "Overlay", win32con.WS_POPUP, 0, 0, screen_width, screen_height, 0,
        0, wnd_class.hInstance, None)
    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
    win32gui.UpdateWindow(hwnd)
    return hwnd


def world_to_screen(entity_pos, player_pos, cam_rot, fov):
    aspect = screen_width / screen_height
    near = 0.1
    far = 1000.0
    fov_rad = math.radians(fov)
    f_val = 1 / math.tan(fov_rad / 2)
    projection_matrix = np.array([
        [f_val / aspect, 0,           0,                           0],
        [0,              f_val,       0,                           0],
        [0,              0,           (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0,              0,          -1,                           0]
    ])
    
    yaw, pitch = map(math.radians, cam_rot)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    
    Rx = np.array([
        [1,  0,   0, 0],
        [0, cp, -sp, 0],
        [0, -sp,  cp, 0],
        [0,  0,   0, 1]
    ])
    Ry = np.array([
        [ cy, 0, sy, 0],
        [  0, 1,  0, 0],
        [-sy, 0, cy, 0],
        [  0, 0,  0, 1]
    ])
    rotation_matrix = Ry @ Rx

    px, py, pz = player_pos
    translation_matrix = np.array([
        [1, 0, 0, -px],
        [0, 1, 0, -py],
        [0, 0, 1, -pz],
        [0, 0, 0,   1]
    ])
    view_matrix = rotation_matrix @ translation_matrix
    
    ex, ey, ez = entity_pos
    entity_homogeneous = np.array([ex, ey, ez, 1.0])
    
    view_coords = view_matrix @ entity_homogeneous
    
    if view_coords[2] <= 0:
        return None, None, None
    
    clip_coords = projection_matrix @ view_coords
    
    if clip_coords[3] != 0:
        ndc = clip_coords[:3] / clip_coords[3]
    else:
        return None, None, None
    
    sx = (ndc[0] + 1.0) * 0.5
    sy = (ndc[1] + 1.0) * 0.5

    
    sx_px = int(sx * screen_width)
    sy_px = int(sy * screen_height)  # Без инверсии

    
    distance = np.linalg.norm(np.array(entity_pos) - np.array(player_pos))
    size = min(7.5, 30.0 / max(1.0, distance))
    return sx_px, sy_px, size

def draw_entities(hwnd):
    try:
        hdc = win32gui.GetDC(0)
        memdc = win32gui.CreateCompatibleDC(hdc)
        bmp = win32gui.CreateCompatibleBitmap(hdc, screen_width, screen_height)
        win32gui.SelectObject(memdc, bmp)
        win32gui.FillRect(memdc, (0, 0, screen_width, screen_height),
                          win32gui.CreateSolidBrush(win32api.RGB(0, 0, 0)))
        global active_entities, player_position, player_camera, player_fov

        with state_lock:
            current_position = player_position
            current_camera = player_camera
            current_fov = player_fov
        with entities_lock:
            ents = [e for e in active_entities.values() if e.etype == 147]
        red_brush = win32gui.CreateSolidBrush(win32api.RGB(185, 113, 255))
        for e in ents:
            sx, sy, scale = world_to_screen((e.x, e.y, e.z), player_position,
                                            player_camera, player_fov)
            if sx is None:
                continue
            w, h = int(25 * scale), int(25 * scale * 1.5)
            l, t = int(sx - w / 2), int(sy - h / 2)
            win32gui.FrameRect(memdc, (l, t, l + w, t + h), red_brush)
            win32gui.SetTextColor(memdc, win32api.RGB(255, 255, 255))
            win32gui.SetBkMode(memdc, win32con.TRANSPARENT)
            win32gui.DrawText(memdc, f"ID: {e.eid}", -1, (l, t - 30, l + w, t - 15),
                              win32con.DT_CENTER)
        blend_func = (win32con.AC_SRC_OVER, 0, 255, win32con.AC_SRC_ALPHA)
        win32gui.UpdateLayeredWindow(hwnd, hdc, None, (screen_width, screen_height),
                                     memdc, (0, 0), 0, blend_func, win32con.ULW_ALPHA)
        win32gui.DeleteObject(bmp)
        win32gui.DeleteDC(memdc)
        win32gui.ReleaseDC(0, hdc)
    except:
        pass


def _decode(fmt, data, offset=0):
    size = struct.calcsize(fmt)
    return struct.unpack_from(fmt, data, offset)[0], offset + size


class DataType:

    class VarInt:

        @staticmethod
        def encode(value: int) -> bytes:
            result = bytearray()
            while value:
                byte = value & 0x7F
                value >>= 7
                result.append(byte | (0x80 if value else 0))
            return bytes(result) or b"\x00"

        @staticmethod
        def decode(data: bytes, offset=0) -> tuple[int, int]:
            result = 0
            shift = 0
            while offset < len(data):
                b = data[offset]
                offset += 1
                result |= (b & 0x7F) << shift
                shift += 7
                if not (b & 0x80):
                    return result, offset
                if shift >= 256:
                    raise ValueError("VarInt слишком большой")
            raise ValueError("VarInt слишком длинный")

    class UnsignedByte:

        @staticmethod
        def encode(value: int) -> bytes:
            return bytes([value])

        @staticmethod
        def decode(data: bytes, offset=0):
            return data[offset], offset + 1

    class Float:

        @staticmethod
        def encode(v: float) -> bytes:
            return struct.pack('>f', v)

        @staticmethod
        def decode(data: bytes, offset=0):
            return _decode('>f', data, offset)

    class Double:

        @staticmethod
        def encode(v: float) -> bytes:
            return struct.pack('>d', v)

        @staticmethod
        def decode(data: bytes, offset=0):
            return _decode('>d', data, offset)

    class Byte:

        @staticmethod
        def encode(v: int) -> bytes:
            return struct.pack('>B', v)

        @staticmethod
        def decode(data: bytes, offset=0):
            return _decode('>B', data, offset)

    class Long:

        @staticmethod
        def encode(v: int) -> bytes:
            return struct.pack(">q", v)

        @staticmethod
        def decode(data: bytes, offset=0):
            return _decode(">q", data, offset)

    class Short:

        @staticmethod
        def encode(v: int) -> bytes:
            return struct.pack('>h', v)

        @staticmethod
        def decode(data: bytes, offset=0):
            return _decode('>h', data, offset)


class State(IntEnum):
    Handshaking = 0
    Status = 1
    Login = 2
    Play = 3


class Direction(IntEnum):
    Clientbound = 0
    Serverbound = 1


class Packet:

    def log(self):
        raise NotImplementedError("Метод log не реализован")


class SetCompressionPacket(Packet):
    packet_id = 0x03

    def __init__(self, threshold: int):
        self.threshold = threshold

    @classmethod
    def decode(cls, payload: bytes) -> 'SetCompressionPacket':
        threshold, _ = DataType.VarInt.decode(payload)
        return cls(threshold)

    def log(self):
        print(f"[+] SetCompression: порог = {self.threshold}")

class SpawnEntityPacket(Packet):
    packet_id = 0x01

    def __init__(self, eid, uuid_, etype, x, y, z, pitch, yaw, head_yaw, data_field, vx,
                 vy, vz):
        self.eid, self.uuid, self.etype = eid, uuid_, etype
        self.x, self.y, self.z = x, y, z
        self.pitch, self.yaw, self.head_yaw = pitch, yaw, head_yaw
        self.data_field, self.vx, self.vy, self.vz = data_field, vx, vy, vz

    @classmethod
    def decode(cls, payload: bytes) -> 'SpawnEntityPacket':
        offset = 0
        eid, offset = DataType.VarInt.decode(payload, offset)
        if len(payload) < offset + 16:
            return
        entity_uuid = uuid.UUID(bytes=payload[offset:offset + 16])
        offset += 16
        etype, offset = DataType.VarInt.decode(payload, offset)
        x, offset = DataType.Double.decode(payload, offset)
        y, offset = DataType.Double.decode(payload, offset)
        z, offset = DataType.Double.decode(payload, offset)
        pitch, offset = DataType.UnsignedByte.decode(payload, offset)
        yaw, offset = DataType.UnsignedByte.decode(payload, offset)
        head_yaw, offset = DataType.UnsignedByte.decode(payload, offset)
        data_field, offset = DataType.VarInt.decode(payload, offset)
        vx, offset = DataType.Short.decode(payload, offset)
        vy, offset = DataType.Short.decode(payload, offset)
        vz, offset = DataType.Short.decode(payload, offset)
        pitch, yaw, head_yaw = (v * 360 / 256 for v in (pitch, yaw, head_yaw))
        return cls(eid, entity_uuid, etype, x, y, z, pitch, yaw, head_yaw, data_field,
                   vx, vy, vz)

    def log(self):
        print(f"[+] SpawnEntity: ID:{self.eid}, UUID:{self.uuid}, Тип:{self.etype}")
        print(f"    Позиция: ({self.x}, {self.y}, {self.z})")
        print(
            f"    Вращение: Pitch:{self.pitch:.2f}, Yaw:{self.yaw:.2f}, Head:{self.head_yaw:.2f}"
        )
        print(
            f"    Data: {self.data_field}, Скорость: ({self.vx}, {self.vy}, {self.vz})")

class RemoveEntitiesPacket(Packet):
    packet_id = 0x47

    def __init__(self, ids: list[int]):
        self.ids = ids

    @classmethod
    def decode(cls, payload: bytes) -> 'RemoveEntitiesPacket':
        offset = 0
        count, offset = DataType.VarInt.decode(payload, offset)
        ids = []
        for _ in range(count):
            eid, offset = DataType.VarInt.decode(payload, offset)
            ids.append(eid)
        return cls(ids)

    def log(self):
        print(f"[+] RemoveEntities: IDs: {self.ids}")

class SetPlayerPositionPacket(Packet):
    packet_id = 0x1C
    def __init__(self, x: float, feet_y: float, z: float, flags: int):
        self.x = x
        self.feet_y = feet_y
        self.z = z
        self.flags = flags

    @classmethod
    def decode(cls, payload: bytes) -> 'SetPlayerPositionPacket':
        offset = 0
        x, offset = DataType.Double.decode(payload, offset)
        feet_y, offset = DataType.Double.decode(payload, offset)
        z, offset = DataType.Double.decode(payload, offset)
        flags, offset = DataType.Byte.decode(payload, offset)
        return cls(x, feet_y, z, flags)

    def log(self):
        on_ground = bool(self.flags & 0x01)
        pushing_against_wall = bool(self.flags & 0x02)
        print(
            f"[+] SetPlayerPositionPacket: X = {self.x}, FeetY = {self.feet_y}, Z = {self.z}"
        )
        print(
            f"    On Ground: {on_ground}, Pushing Against Wall: {pushing_against_wall}")

class SetPlayerPositionAndRotationPacket(Packet):
    packet_id = 0x1D

    def __init__(self, x: float, feet_y: float, z: float, yaw: float, pitch: float, flags: int):
        self.x = x
        self.feet_y = feet_y
        self.z = z
        self.yaw = yaw
        self.pitch = pitch
        self.flags = flags

    @classmethod
    def decode(cls, payload: bytes) -> 'SetPlayerPositionAndRotationPacket':
        offset = 0
        x, offset = DataType.Double.decode(payload, offset)
        feet_y, offset = DataType.Double.decode(payload, offset)
        z, offset = DataType.Double.decode(payload, offset)
        yaw, offset = DataType.Float.decode(payload, offset)
        pitch, offset = DataType.Float.decode(payload, offset)
        flags, offset = DataType.Byte.decode(payload, offset)
        return cls(x, feet_y, z, yaw, pitch, flags)

    def log(self):
        on_ground = bool(self.flags & 0x01)
        pushing_against_wall = bool(self.flags & 0x02)
        print(
            f"[+] SetPlayerPositionAndRotation: X = {self.x}, FeetY = {self.feet_y}, Z = {self.z}, Yaw = {self.yaw}, Pitch = {self.pitch}"
        )
        print(f"    On Ground: {on_ground}, Pushing Against Wall: {pushing_against_wall}")

class UpdateEntityPositionPacket(Packet):
    packet_id = 0x2F

    def __init__(self, eid, dx, dy, dz, on_ground: bool):
        self.eid = eid
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.on_ground = on_ground

    @classmethod
    def decode(cls, payload: bytes) -> 'UpdateEntityPositionPacket':
        offset = 0
        eid, offset = DataType.VarInt.decode(payload, offset)
        dx, offset = DataType.Short.decode(payload, offset)
        dy, offset = DataType.Short.decode(payload, offset)
        dz, offset = DataType.Short.decode(payload, offset)
        on_ground, offset = DataType.Byte.decode(payload, offset)
        dx = dx / 4096.0
        dy = dy / 4096.0
        dz = dz / 4096.0
        on_ground = bool(on_ground)
        return cls(eid, dx, dy, dz, on_ground)

    def log(self):
        print(
            f"[+] UpdateEntityPosition: ID:{self.eid} Δx:{self.dx}, Δy:{self.dy}, Δz:{self.dz}, OnGround:{self.on_ground}"
        )

class UpdateEntityPositionAndRotationPacket(Packet):
    packet_id = 0x30

    def __init__(self, eid, dx, dy, dz, yaw, pitch, on_ground: bool):
        self.eid = eid
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.yaw = yaw
        self.pitch = pitch
        self.on_ground = on_ground

    @classmethod
    def decode(cls, payload: bytes) -> 'UpdateEntityPositionAndRotationPacket':
        offset = 0
        eid, offset = DataType.VarInt.decode(payload, offset)
        dx, offset = DataType.Short.decode(payload, offset)
        dy, offset = DataType.Short.decode(payload, offset)
        dz, offset = DataType.Short.decode(payload, offset)
        yaw, offset = DataType.UnsignedByte.decode(payload, offset)
        pitch, offset = DataType.UnsignedByte.decode(payload, offset)
        on_ground, offset = DataType.Byte.decode(payload, offset)

        dx = dx / 4096.0
        dy = dy / 4096.0
        dz = dz / 4096.0

        yaw = yaw * 360 / 256
        pitch = pitch * 360 / 256
        on_ground = bool(on_ground)
        return cls(eid, dx, dy, dz, yaw, pitch, on_ground)

    def log(self):
        print(f"[+] UpdateEntityPositionAndRotation: ID:{self.eid} Δx:{self.dx}, Δy:{self.dy}, Δz:{self.dz}, Yaw:{self.yaw}, Pitch:{self.pitch}, OnGround:{self.on_ground}")

class SetPlayerRotationPacket(Packet):
    packet_id = 0x1E

    def __init__(self, yaw: float, pitch: float, flags: int):
        self.yaw = self.normalize_yaw(yaw)
        self.pitch = pitch
        self.flags = flags

    @classmethod
    def decode(cls, payload: bytes) -> 'SetPlayerRotationPacket':
        offset = 0
        yaw, offset = DataType.Float.decode(payload, offset)
        pitch, offset = DataType.Float.decode(payload, offset)
        flags, offset = DataType.Byte.decode(payload, offset)
        return cls(yaw, pitch, flags)

    @staticmethod
    def normalize_yaw(yaw: float) -> float:
        """Нормализует yaw в диапазон [-180, 180]"""
        return ((yaw + 180) % 360) - 180

    def log(self):
        on_ground = bool(self.flags & 0x01)
        pushing_against_wall = bool(self.flags & 0x02)
        print(f"[+] SetPlayerRotationPacket: Yaw = {self.yaw}, Pitch = {self.pitch}")
        print(
            f"    On Ground: {on_ground}, Pushing Against Wall: {pushing_against_wall}")


class PacketFactory:
    packet_decoders = {
        SetCompressionPacket.packet_id: SetCompressionPacket.decode,
        SpawnEntityPacket.packet_id: SpawnEntityPacket.decode,
        RemoveEntitiesPacket.packet_id: RemoveEntitiesPacket.decode,
        SetPlayerPositionPacket.packet_id: SetPlayerPositionPacket.decode,
        SetPlayerPositionAndRotationPacket.packet_id: SetPlayerPositionAndRotationPacket.decode,
        UpdateEntityPositionPacket.packet_id: UpdateEntityPositionPacket.decode,
        UpdateEntityPositionAndRotationPacket.packet_id: UpdateEntityPositionAndRotationPacket.decode,
        SetPlayerRotationPacket.packet_id: SetPlayerRotationPacket.decode,
    }

    @staticmethod
    def decode_packet(payload: bytes) -> Packet:
        pid, offset = DataType.VarInt.decode(payload)
        data = payload[offset:]
        return PacketFactory.packet_decoders.get(pid, lambda x: (None, None))(data)


class MinecraftProxy:

    def __init__(self, local_host, local_port, server_host, server_port):
        self.local_host, self.local_port = local_host, local_port
        self.server_host, self.server_port = server_host, server_port
        self.compression_enabled = False
        self.compression_threshold = None

    def start(self):
        with socket.socket() as srv:
            srv.bind((self.local_host, self.local_port))
            srv.listen(5)
            print(f"[ i ] Proxy started -> {self.local_host}:{self.local_port}")

            while True:
                client, addr = srv.accept()
                threading.Thread(target=self.handle_connection,
                                 args=(client, ),
                                 daemon=True).start()

    def handle_connection(self, client):
        try:
            with socket.socket() as server:
                server.connect((self.server_host, self.server_port))
                print(f"[ + ] Connected to server -> {self.server_host}:{self.server_port}")

                c = threading.Thread(target=self.forward_client,
                                     args=(client, server),
                                     daemon=True)
                s = threading.Thread(target=self.forward_server,
                                     args=(server, client),
                                     daemon=True)

                c.start()
                s.start()
                c.join()
                s.join()
        except Exception as e:
            print(f"[ - ] Connection error -> {e}")
        finally:
            server.close()
            client.close()

    def read_packet(self, sock):
        try:
            plen, plen_bytes = self.read_varint(sock)
            pdata = self.read_exactly(sock, plen)
            return plen_bytes + pdata if pdata else None
        except Exception:
            return None

    def read_exactly(self, sock, n):
        data = bytearray()
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def read_varint(self, sock):
        raw = bytearray()
        val = 0
        shift = 0
        while True:
            byte = sock.recv(1)
            if not byte:
                break
            raw.append(byte[0])
            val |= (byte[0] & 0x7F) << shift
            if not (byte[0] & 0x80):
                break
            shift += 7
        return val, bytes(raw)

    def forward_client(self, src, dst):
        try:
            while (packet := self.read_packet(src)):
                self.process_client_packet(packet)
                dst.sendall(packet)
        except Exception as e:
            pass

    def process_client_packet(self, packet: bytes):
        global player_position
        global player_camera
        try:
            _, offset = DataType.VarInt.decode(packet)
        except Exception as e:
            return
        data = packet[offset:]
        if self.compression_enabled:
            try:
                ulen, off = DataType.VarInt.decode(data)
            except Exception as e:
                return
            if ulen:
                try:
                    packet_payload = zlib.decompress(data[off:])
                except Exception as e:
                    return
            else:
                packet_payload = data[off:]
        else:
            packet_payload = data
        try:
            pkt = PacketFactory.decode_packet(packet_payload)
            if isinstance(pkt, SetPlayerPositionPacket):
                with state_lock:
                    player_position = (pkt.x, pkt.feet_y, pkt.z)
            if isinstance(pkt, SetPlayerRotationPacket):
                with state_lock:
                    player_camera = (pkt.yaw, pkt.pitch)
            if isinstance(pkt, SetPlayerPositionAndRotationPacket):
                with state_lock:
                    player_position = (pkt.x, pkt.feet_y, pkt.z)
                    player_camera = (pkt.yaw, pkt.pitch)
        except Exception as e:
            pass

    def forward_server(self, src, dst):
        try:
            while (packet := self.read_packet(src)):
                self.process_server_packet(packet)
                dst.sendall(packet)
        except Exception as e:
            pass

    def process_server_packet(self, packet: bytes):
        try:
            _, offset = DataType.VarInt.decode(packet)
            data = packet[offset:]

            if self.compression_enabled:
                ulen, off = DataType.VarInt.decode(data)
                if ulen:
                    try:
                        packet_payload = zlib.decompress(data[off:])
                    except Exception as e:
                        return
                else:
                    packet_payload = data[off:]
            else:
                packet_payload = data

            pkt = PacketFactory.decode_packet(packet_payload)

            if isinstance(pkt, SetCompressionPacket):
                self.compression_enabled = pkt.threshold >= 0
                self.compression_threshold = pkt.threshold
            elif isinstance(pkt, SpawnEntityPacket):
                with entities_lock:
                    active_entities[pkt.eid] = pkt
            elif isinstance(pkt, RemoveEntitiesPacket):
                with entities_lock:
                    for eid in pkt.ids:
                        active_entities.pop(eid, None)
            elif isinstance(pkt, UpdateEntityPositionPacket):
                with entities_lock:
                    entity = active_entities.get(pkt.eid)
                    if entity:
                        entity.x += pkt.dx
                        entity.y += pkt.dy
                        entity.z += pkt.dz
                    else:
                        pass
        except Exception as e:
            pass


def start_drawing():
    hwnd = create_fullscreen_window()
    ctypes.windll.user32.SetTimer(hwnd, 1, 16, None)
    win32gui.PumpMessages()

if __name__ == "__main__":
    LOCAL_HOST = "127.0.0.1"
    LOCAL_PORT = 25565
    SERVER_HOST = "<server_ip>"
    SERVER_PORT = 25565

    proxy_thread = threading.Thread(target=lambda: MinecraftProxy(
        LOCAL_HOST, LOCAL_PORT, SERVER_HOST, SERVER_PORT).start(),
                                    daemon=True)
    proxy_thread.start()

    start_drawing()
