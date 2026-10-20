"""Read-only Google Drive source, cancellable downloads and managed local cleanup."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import task_control as tasks
from project_store import atomic_json, digest, fingerprint

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FIELDS = 'id,name,size,md5Checksum,modifiedTime,version,mimeType,capabilities(canDownload)'


def folder_id(value):
    value = value.strip()
    if not value:
        return None
    if '://' in value:
        parsed = urlparse(value)
        if parsed.hostname not in ('drive.google.com', 'www.drive.google.com'):
            raise ValueError('Нужна ссылка на папку Google Drive.')
        match = re.search(r'/folders/([A-Za-z0-9_-]+)', parsed.path)
        if not match:
            raise ValueError('Нужна ссылка вида drive.google.com/drive/folders/…')
        return match.group(1)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('Некорректный ID папки.')
    return value


class DriveSource:
    def __init__(self, folder, service=None):
        self.root = Path(folder) / 'moto_drive'
        self.root.mkdir(parents=True, exist_ok=True)
        self.downloads = self.root / 'downloads'
        self.registry = self.root / 'sources.json'
        self.settings = self.root / 'settings.json'
        self.token = self.root / 'token.json'
        self.service = service
        self.entries = json.loads(self.registry.read_text(encoding='utf-8')) if self.registry.exists() else {}
        try:
            self.auto_cleanup = bool(json.loads(self.settings.read_text(encoding='utf-8')).get('auto_cleanup',True))
        except FileNotFoundError:
            self.auto_cleanup = True
        except (ValueError,OSError):
            self.auto_cleanup = False

    def save(self):
        atomic_json(self.registry, self.entries)

    def set_auto_cleanup(self, enabled):
        atomic_json(self.settings,{'auto_cleanup':bool(enabled)})
        self.auto_cleanup=bool(enabled)

    def api(self):
        if self.service is not None:
            return self.service
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from google_auth_httplib2 import AuthorizedHttp
            from googleapiclient.discovery import build
            import httplib2
        except ImportError as exc:
            raise RuntimeError('Запустите Установить_Google_Drive.bat.') from exc
        if not self.token.exists():
            raise RuntimeError('Сначала нажмите «ПОДКЛЮЧИТЬ GOOGLE».')
        creds = Credentials.from_authorized_user_file(str(self.token), SCOPES)
        if not creds.has_scopes(SCOPES):
            raise RuntimeError('Повторно подключите Google с доступом на чтение видео.')
        if not creds.valid:
            if not creds.refresh_token:
                raise RuntimeError('Повторно подключите Google.')
            try:
                creds.refresh(Request())
            except Exception as exc:
                raise RuntimeError('Не удалось обновить вход Google. Подключите аккаунт заново.') from exc
            self.write_token(creds.to_json())
        self.service = build('drive', 'v3', http=AuthorizedHttp(creds, http=httplib2.Http(timeout=30)),
                             cache_discovery=False)
        return self.service

    def write_token(self, value):
        atomic_json(self.token, json.loads(value))
        try:
            os.chmod(self.token, 0o600)
        except OSError:
            pass

    def connect(self, client_file):
        # The child owns the loopback listener; STOP terminates it immediately.
        result = tasks.run([sys.executable, str(Path(__file__).resolve()), '--authorize',
                            str(self.root.parent), str(Path(client_file).resolve())], capture_output=True)
        if result.returncode or not self.token.exists():
            raise RuntimeError('Вход Google не завершён. Проверьте Desktop OAuth JSON и тестового пользователя в Google Cloud.')
        self.service = None

    def list_videos(self, folder=None, recursive=False):
        api = self.api()
        folder = folder_id(folder or '')
        queue, visited, videos = [folder], set(), []
        while queue:
            tasks.check()
            parent = queue.pop(0)
            if parent in visited:
                continue
            visited.add(parent)
            if len(visited) > 1000:
                raise ValueError('Слишком много вложенных папок. Выберите папку с нужной поездкой.')
            query = 'trashed = false'
            if parent:
                query += f" and '{parent}' in parents"
            else:
                query += " and (mimeType = 'video/mp4' or name contains '.mp4' or name contains '.MP4')"
            page = None
            while True:
                tasks.check()
                response = api.files().list(q=query, fields=f'nextPageToken,files({FIELDS})', pageSize=1000,
                    pageToken=page, supportsAllDrives=True, includeItemsFromAllDrives=True).execute(num_retries=1)
                for item in response.get('files', []):
                    if item.get('mimeType') == 'application/vnd.google-apps.folder':
                        if recursive:
                            queue.append(item['id'])
                    elif item['name'].lower().endswith('.mp4'):
                        videos.append(item)
                page = response.get('nextPageToken')
                if not page:
                    break
        return sorted(videos, key=lambda x: x['name'].casefold())

    def add(self, metadata):
        fid = metadata['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', fid):
            raise ValueError('Некорректный ID Google Drive.')
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', metadata['name']).strip(' .') or 'video.mp4'
        if Path(name).stem.upper() in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}:
            name = '_'+name
        if len(name)>100:name=Path(name).stem[:94]+'.mp4'
        path = (self.downloads / fid / name).resolve()
        previous=self.entries.get(str(path),{})
        def identity(value):
            keys=('md5Checksum','size') if value.get('md5Checksum') else ('version','size')
            return {key:value.get(key) for key in keys}
        # Keep old project paths bound to the version they actually describe.
        # Re-adding a changed cloud file must not replace its previous entry.
        if previous and identity(previous.get('metadata',{})) != identity(metadata):
            revision='revision-'+digest(identity(metadata))[:16]
            path=(self.downloads/fid/revision/name).resolve()
            previous=self.entries.get(str(path),{})
        entry={'metadata':metadata,'path':str(path)}
        old_metadata=previous.get('metadata',{})
        if metadata.get('md5Checksum') and metadata['md5Checksum']==old_metadata.get('md5Checksum'):
            if previous.get('download_fingerprint'):entry['download_fingerprint']=previous['download_fingerprint']
        self.entries[str(path)] = entry
        self.save()
        return path

    def entry(self, path):
        return self.entries.get(str(Path(path).resolve()))

    def ensure(self, path, on_progress=lambda f: None):
        tasks.check()
        path = Path(path).resolve()
        entry = self.entry(path)
        if not entry:
            if path.is_file():
                return path
            raise FileNotFoundError('Исходник удалён. Сведения о событии сохранены, но для кадров и экспорта нужен MP4.')
        if not path.is_relative_to(self.downloads.resolve()):
            raise ValueError('Недопустимый путь загрузки Google Drive.')
        if path.is_file():
            if entry.get('download_fingerprint') == fingerprint(path):
                return path
            raise ValueError('Временный MP4 изменён или заменён. Переместите его в другую папку, прежде чем загружать оригинал снова.')
        metadata = entry['metadata']
        current = self.api().files().get(fileId=metadata['id'], fields=FIELDS, supportsAllDrives=True).execute(num_retries=1)
        for key in ('md5Checksum', 'size') if metadata.get('md5Checksum') else ('version','size'):
            if metadata.get(key) and current.get(key) != metadata[key]:
                raise ValueError('Запись на Google Drive изменилась. Добавьте её в очередь заново; прежние события сохранены.')
        if not current.get('capabilities', {}).get('canDownload', False):
            raise ValueError('Владелец Google Drive запретил загрузку этого видео.')
        from googleapiclient.http import MediaIoBaseDownload
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix+'.part')
        try:
            request = self.api().files().get_media(fileId=metadata['id'], supportsAllDrives=True)
            with partial.open('wb') as stream:
                download = MediaIoBaseDownload(stream, request, chunksize=4*1024*1024)
                done = False
                while not done:
                    tasks.check()
                    status, done = download.next_chunk(num_retries=1)
                    if status:
                        on_progress(status.progress())
            tasks.check()
            if current.get('size') and partial.stat().st_size != int(current['size']):
                raise ValueError('Видео загрузилось не полностью.')
            if current.get('md5Checksum'):
                checksum = hashlib.md5()
                with partial.open('rb') as stream:
                    while block := stream.read(1024*1024):
                        tasks.check()
                        checksum.update(block)
                if checksum.hexdigest() != current['md5Checksum']:
                    raise ValueError('Контрольная сумма загруженного видео не совпала.')
            os.replace(partial, path)
            entry['download_fingerprint'] = fingerprint(path)
            self.save()
            return path
        finally:
            partial.unlink(missing_ok=True)

    def release(self, path, archive):
        tasks.check()
        path = Path(path).resolve()
        entry = self.entry(path)
        if not entry or not path.is_relative_to(self.downloads.resolve()) or not path.is_file():
            return False
        if not archive.can_release(path) or entry.get('download_fingerprint') != fingerprint(path):
            return False
        tasks.check()
        path.unlink()
        # Optional LRFs are not managed by this downloader and are never deleted here.
        return True


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--authorize':
        raise SystemExit(2)
    from google_auth_oauthlib.flow import InstalledAppFlow
    client = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8-sig'))
    if 'installed' not in client:
        raise SystemExit('Нужен OAuth-клиент типа Desktop app.')
    flow = InstalledAppFlow.from_client_config(client, SCOPES)
    credentials = flow.run_local_server(host='127.0.0.1', port=0, timeout_seconds=180,
        authorization_prompt_message='Завершите вход в открывшемся браузере.',
        success_message='Google подключён. Это окно можно закрыть.')
    DriveSource(sys.argv[2]).write_token(credentials.to_json())
