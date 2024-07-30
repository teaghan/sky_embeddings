import logging

logger = logging.getLogger()


class SharedMemoryRegistry:
    _registered_objects = []

    @classmethod
    def register(cls, obj):
        cls._registered_objects.append(obj)

    @classmethod
    def unregister(cls, obj):
        cls._registered_objects.remove(obj)

    @classmethod
    def cleanup_all(cls):
        logger.info('Cleaning up shared memory..')
        for obj in cls._registered_objects:
            obj.cleanup()


class H5FileRegistry:
    _registered_files = []

    @classmethod
    def register(cls, h5_file):
        cls._registered_files.append(h5_file)

    @classmethod
    def unregister(cls, h5_file):
        cls._registered_files.remove(h5_file)

    @classmethod
    def cleanup_all(cls):
        for h5_file in cls._registered_files:
            if h5_file and h5_file.id:  # Check if file is still open
                logger.info(f'Closing H5 file: {h5_file.filename}')
                h5_file.close()
        cls._registered_files.clear()
