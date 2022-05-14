package tripos.partIIB.gestureflow;

public class Rectangle {
    Point top_right;
    Point bot_left;
    float width;
    float height;
    float area;
    float diagonal;

    public Rectangle(Point top_right, Point bot_left){
        this.top_right = top_right;
        this.bot_left = bot_left;
        this.width = Math.abs(top_right.getX() - bot_left.getX());
        this.height = Math.abs(top_right.getY() - bot_left.getY());
        this.area = width * height;
        this.diagonal = (float) Math.pow(Math.pow(width, 2) + Math.pow(height, 2), 0.5);
    }

    public Rectangle(float min_x, float min_y, float max_x, float max_y){
        this.top_right = new Point(min_x, min_y);
        this.bot_left = new Point(max_x, max_y);
        this.width = Math.abs(top_right.getX() - bot_left.getX());
        this.height = Math.abs(top_right.getY() - bot_left.getY());
        this.area = width * height;
        this.diagonal = (float) Math.pow(Math.pow(width, 2) + Math.pow(height, 2), 0.5);
    }

    public Float getWidth() {
        return width;
    }

    public Float getHeight() {
        return height;
    }

    public Float getArea() {
        return area;
    }

    public Float getDiagonal() { return diagonal; }

    public void setTopRight(Point top_right) {
        this.top_right = top_right;
        this.width = Math.abs(top_right.getX() - bot_left.getX());
        this.height = Math.abs(top_right.getY() - bot_left.getY());
        this.area = width * height;
        this.diagonal = (float) Math.pow(Math.pow(width, 2) + Math.pow(height, 2), 0.5);
    }

    public void setBotLeft(Point bot_left) {
        this.bot_left = bot_left;
        this.width = Math.abs(top_right.getX() - bot_left.getX());
        this.height = Math.abs(top_right.getY() - bot_left.getY());
        this.area = width * height;
        this.diagonal = (float) Math.pow(Math.pow(width, 2) + Math.pow(height, 2), 0.5);
    }

}
