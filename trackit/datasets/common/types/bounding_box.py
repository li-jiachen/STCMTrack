import enum


class BoundingBoxCoordinateSystem(enum.Enum):
    Discrete = enum.auto() 
    Continuous = enum.auto()


class BoundingBoxFormat(enum.Enum):
    
    XYWH = enum.auto()
    XYXY = enum.auto()
    Polygon = enum.auto()
