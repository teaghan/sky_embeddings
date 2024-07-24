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
        for obj in cls._registered_objects:
            obj.cleanup()
