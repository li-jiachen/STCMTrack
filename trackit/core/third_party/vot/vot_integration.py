

import os
import collections
import numpy as np

try:
    import trax
except ImportError:
    raise Exception('TraX support not found. Please add trax module to Python path.')

if trax._ctypes.trax_version().decode("ascii") < "4.0.0":
    raise ImportError('TraX version 4.0.0 or newer is required.')

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
Point = collections.namedtuple('Point', ['x', 'y'])
Polygon = collections.namedtuple('Polygon', ['points'])
Empty = collections.namedtuple('Empty', [])

class VOT(object):
    
    def __init__(self, region_format, channels=None, multiobject: bool = None):
        
        assert(region_format in [trax.Region.RECTANGLE, trax.Region.POLYGON, trax.Region.MASK])

        if multiobject is None:
            multiobject = os.environ.get('VOT_MULTI_OBJECT', '0') == '1'

        if channels is None:
            channels = ['color']
        elif channels == 'rgbd':
            channels = ['color', 'depth']
        elif channels == 'rgbt':
            channels = ['color', 'ir']
        elif channels == 'ir':
            channels = ['ir']
        else:
            raise Exception('Illegal configuration {}.'.format(channels))

        self._trax = trax.Server([region_format], [trax.Image.PATH], channels, metadata=dict(vot="python"), multiobject=multiobject)

        request = self._trax.wait()
        assert(request.type == 'initialize')

        self._objects = []

        assert len(request.objects) > 0 and (multiobject or len(request.objects) == 1)

        for object, _ in request.objects:
            if isinstance(object, trax.Polygon):
                self._objects.append(Polygon([Point(x[0], x[1]) for x in object]))
            elif isinstance(object, trax.Mask):
                self._objects.append(object.array(True))
            else:
                self._objects.append(Rectangle(*object.bounds()))

        self._image = [x.path() for k, x in request.image.items()]
        if len(self._image) == 1:
            self._image = self._image[0]

        self._multiobject = multiobject

        self._trax.status(request.objects)

    def region(self):
        

        assert not self._multiobject

        return self._objects[0]

    def objects(self):
        

        return self._objects

    def report(self, status, confidence = None):
        

        def convert(region):
            
            
            if region is None: return trax.Rectangle.create(0, 0, 0, 0)
            assert isinstance(region, (Empty, Rectangle, Polygon, np.ndarray)), f"unexpected data type, got {type(region)}"
            if isinstance(region, Empty):
                return trax.Rectangle.create(0, 0, 0, 0)
            elif isinstance(region, Polygon):
                return trax.Polygon.create([(x.x, x.y) for x in region.points])
            elif isinstance(region, np.ndarray):
                return trax.Mask.create(region)
            else:
                return trax.Rectangle.create(region.x, region.y, region.width, region.height)

        if not self._multiobject:
            properties = {}
            if not confidence is None:
                properties['confidence'] = confidence
            status = [(convert(status), properties)]
        else:
            assert isinstance(status, (list, tuple))
            status = [(convert(x), {}) for x in status]

        self._trax.status(status, {})

    def frame(self):
        
        if hasattr(self, "_image"):
            image = self._image
            del self._image
            return image

        request = self._trax.wait()

        
        assert request.objects is None or len(request.objects) == 0

        if request.type == 'frame':
            image = [x.path() for k, x in request.image.items()]
            if len(image) == 1:
                return image[0]
            return image
        else:
            return None

    def quit(self):
        
        if hasattr(self, '_trax'):
            self._trax.quit()

    def __del__(self):
        
        self.quit()

class VOTManager(object):
    

    def __init__(self, factory, region_format, channels=None):
        
        self._handle = VOT(region_format, channels, multiobject=True)
        self._factory = factory

    def run(self):
        
        objects = self._handle.objects()

        
        image = self._handle.frame()
        if not image:
            return

        trackers = [self._factory(image, object) for object in objects]

        while True:

            image = self._handle.frame()
            if not image:
                break

            status = [tracker(image) for tracker in trackers]

            self._handle.report(status)