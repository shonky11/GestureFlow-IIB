package tripos.partIIB.gestureflow;

public class Point {
    private Float X;
    private Float Y;

    public Point(Float x, Float y) {
        X = x;
        Y = y;
    }

    public Float getX() {
        return X;
    }

    public void setX(Float x) {
        this.X = x;
    }

    public Float getY() {
        return Y;
    }

    public void setY(Float y) {
        this.Y = y;
    }

    public Float[] getPoint() {
        Float[] point = {X, Y};
        return point;
    }

    public void setPoint(Float x, Float y) {
        this.X = x;
        this.Y = y;
    }

}
