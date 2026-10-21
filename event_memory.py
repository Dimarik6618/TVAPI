"""Freeze candidate evidence before a managed source can be removed."""
import zlib

from event_archive import EventArchive
from video_features import VideoFeatureExtractor
import task_control as tasks


def remember(path, duration, data, clips, folder, runtime, pool, selector=None, origin=None,
             on_progress=lambda f: None, freeze_features=True, on_status=lambda text: None):
    archive = EventArchive(folder)
    sid = archive.register(path, duration, origin, data)
    key = 'memory-features:'+runtime.profile
    extractor = selector.extractor if selector else pool.get(key)
    failure, ids = '', []
    for i, clip in enumerate(clips):
        tasks.check()
        on_status(f'Память событий: {i+1}/{len(clips)}')
        if not clip.embedding:
            try:
                saved = archive.feature(sid, clip.start, clip.end)
                if saved is not None:
                    clip.embedding = saved[0].tolist()
            except (ValueError, TypeError, zlib.error):
                # An invalid cached vector must be recomputed, never marked ready.
                pass
        if freeze_features and not clip.embedding and not failure:
            try:
                if extractor is None:
                    extractor = VideoFeatureExtractor(__import__('moto_shorts').ffmpeg_exe(), folder,
                                                      runtime, on_status=on_status, archive=archive)
                    advanced = next((value for value in pool.values() if getattr(value,'video_model',None) is not None),None)
                    if advanced:
                        import transformers
                        from video_features import FEATURE_SPEC
                        if transformers.__version__ == FEATURE_SPEC['transformers']:
                            extractor.model,extractor.processor,extractor.torch = advanced.video_model,advanced.video_processor.image_processor,advanced.torch
                            extractor.runtime = advanced.runtime
                    pool[key] = extractor
                on_status(f'Признаки для обучения: {i+1}/{len(clips)}')
                has_status = hasattr(extractor, 'on_status')
                previous_status = getattr(extractor, 'on_status', None)
                try:
                    if has_status:
                        extractor.on_status = on_status
                    clip.embedding = extractor.extract(path, clip.start, clip.end).tolist()
                finally:
                    if has_status:
                        extractor.on_status = previous_status
            except Exception as exc:
                failure = str(exc)
        clip.details['memory'] = 'ready' if clip.embedding else 'analysis_only'
        ids.append(archive.save_clip(sid, clip))
        on_progress((i+1)/max(1,len(clips)))
    tasks.check()
    archive.finish(sid, ids)
    return failure
