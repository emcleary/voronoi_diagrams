from voronoi_diagrams.src.point import Point


class Edge[P: Point]:
    """
    An edge joining 2 n-dimensional points

    Attributes:
        _p0 (Point): The source point of the edge
        _p1 (Point): The destination point of the edge
    """

    def __init__(self, source: P, destination: P):
        """
        Creates and edge object

        Args:
            p0 (Point): The source point
            p1 (Point): The destination point
        """
        self._src = source
        self._dest = destination

    @property
    def src(self) -> P:
        """
        Gets the source point

        Return:
            Point: The source point
        """
        return self._src

    @property
    def dest(self) -> P:
        """
        Gets the destination point

        Return:
            Point: The destination point
        """
        return self._dest

    def __str__(self) -> str:
        return f"{self.src} -> {self.dest}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.src} {self.dest})"
