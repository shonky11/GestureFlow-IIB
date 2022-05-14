package tripos.partIIB.gestureflow;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.os.Build;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.firestore.DocumentReference;
import com.google.firebase.firestore.FirebaseFirestore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CustomView extends View {
    //drawing path
    public Path drawPath;
    //defines what to draw
    public Paint canvasPaint;
    //defines how to draw
    public Paint drawPaint;
    public Paint drawPaint2;
    //initial color
    public int paintColor = 0xFF85B09A;
    public int paintColor2 = 0xAA85B09A;
    //canvas - holding pen, holds your drawings
    //and transfers them to the view
    public Canvas drawCanvas;
    //canvas bitmap
    public Bitmap canvasBitmap;
    //brush size
    public float currentBrushSize, lastBrushSize;

    public PreProcessor preprocessor;

    public Boolean training = true;

    float mX, mY, TOUCH_TOLERANCE;

    public ArrayList<Path> paths = new ArrayList<Path>();

    public CustomView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        this.setBackgroundColor(Color.parseColor("#1f1f2b"));
        //apply bitmap to graphic to start drawing.
        init();
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        //create canvas of certain device size.
        super.onSizeChanged(w, h, oldw, oldh);
        Toast.makeText(this.getContext(), "hehe", Toast.LENGTH_SHORT).show();

        //create Bitmap of certain w,h
        canvasBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);

        //apply bitmap to graphic to start drawing.
        drawCanvas = new Canvas(canvasBitmap);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float touchX = event.getX();
        float touchY = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                touch_start(touchX, touchY);
                invalidate();
                break;
            case MotionEvent.ACTION_MOVE:
                touch_move(touchX, touchY);
                invalidate();
                break;
            case MotionEvent.ACTION_UP:
                touch_up();
                invalidate();
                break;
            default:
                return false;
        }
        return true;
    }

    private void touch_start(float x, float y) {
        drawPath.reset();
        drawPath.moveTo(x, y);
        mX = x;
        mY = y;
    }

    private void touch_move(float x, float y) {
        float dx = Math.abs(x - mX);
        float dy = Math.abs(y - mY);
        if (dx >= TOUCH_TOLERANCE || dy >= TOUCH_TOLERANCE) {
            drawPath.quadTo(mX, mY, (x + mX) / 2, (y + mY) / 2);
            mX = x;
            mY = y;
        }
    }

    private void touch_up() {
        drawPath.lineTo(mX, mY);
        drawCanvas.drawPath(drawPath, drawPaint);
        paths.add(drawPath);
        drawPath = new Path();

    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int width = canvas.getWidth();
        int height = canvas.getHeight();
        float[] sum = new float[16];
        int[] count = new int[16];
        int prev = 0;
//        int workable_width = (int) (0.8 * width);
//        int workable_height = (int) (0.8 * height);

        if (!drawPath.isEmpty()) {
//            canvas.drawPath(p, drawPaint);

            float[] collection = new float[16];

            for (int i = 0; i < 16; i++) {
                ArrayList<Point> points = preprocessor.Approx2Gest(drawPath.approximate(0.01F));
                Rectangle temp = preprocessor.dynamic_bounding_box(points, canvas.getWidth(), canvas.getHeight(), i);
                Rectangle rect_tight_inp = preprocessor.bounding_box(points);
                //canvas.drawRect(Math.round(temp.top_right.getX()), Math.round(temp.top_right.getY()), Math.round(temp.bot_left.getX()), Math.round(temp.bot_left.getY()), drawPaint);

                float exp_path_length = (float) (preprocessor.Path2Bound(i) * temp.diagonal);
                float tru_path_length = (float) (preprocessor.path_length(points));
                int valid = Math.min(Math.round(30 / (exp_path_length / tru_path_length)), 30);

                ArrayList<Point> resampled = preprocessor.resample(points, valid);
                ArrayList<Point> normalized = preprocessor.normalize(resampled, temp);
                ArrayList<Point> offsets = preprocessor.offsets(normalized);

                float[] pred = preprocessor.model_predict(offsets);

                for (int z = 0; z < 16; z++) {
                    collection[z] += pred[z];
                }
            }

            ArrayList<Integer> predictions = preprocessor.predict(collection);

            for(Integer i : predictions){
                ArrayList<Point> points = preprocessor.Approx2Gest(drawPath.approximate(0.01F));
                Rectangle temp = preprocessor.dynamic_bounding_box(points, canvas.getWidth(), canvas.getHeight(), i);
                Rectangle rect_tight_inp = preprocessor.bounding_box(points);
                //canvas.drawRect(Math.round(temp.top_right.getX()), Math.round(temp.top_right.getY()), Math.round(temp.bot_left.getX()), Math.round(temp.bot_left.getY()), drawPaint);

                float exp_path_length = (float) (preprocessor.Path2Bound(i) * temp.diagonal);
                float tru_path_length = (float) (preprocessor.path_length(points));
                int valid = Math.min(Math.round(30 / (exp_path_length / tru_path_length)), 30);

                if (valid > prev) {
                    prev = valid + 2;
                    ArrayList<Point> resampled = preprocessor.resample(points, valid);
                    ArrayList<Point> normalized = preprocessor.normalize(resampled, temp);
                    ArrayList<Point> offsets = preprocessor.offsets(normalized);
                    ArrayList<Point> gen = preprocessor.model_generate(offsets, getContext(), i, new Point(500F, 500F));
                    ArrayList<Point> reconstructed = preprocessor.reconstruct(gen, new Point(0.0F, 0.0F));
                    Rectangle tight_rect_output = preprocessor.bounding_box(preprocessor.reconstruct(offsets, new Point(0.0F, 0.0F)));
                    ArrayList<Point> resized = preprocessor.resize(reconstructed, rect_tight_inp.getWidth(), rect_tight_inp.getHeight(), tight_rect_output);
                    Point start = new Point(points.get(0).getX(), points.get(0).getY());
                    ArrayList<Point> translated = preprocessor.overlay(resized, start);
                    Log.d("myTag", String.valueOf(gen.size()));
                    canvas.drawPath(preprocessor.draw(translated), drawPaint2);
                }
            }
            canvas.drawPath(drawPath, drawPaint);
        }
//        canvas.drawPath(drawPath, drawPaint);
    }

    private void init() {
        currentBrushSize = getResources().getInteger(R.integer.medium_size);
        lastBrushSize = currentBrushSize;
        drawPath = new Path();
        drawPaint = new Paint();
        drawPaint.setColor(paintColor);
        drawPaint.setAntiAlias(true);
        drawPaint.setStrokeWidth(currentBrushSize);
        drawPaint.setStyle(Paint.Style.STROKE);
        drawPaint.setStrokeJoin(Paint.Join.ROUND);
        drawPaint.setStrokeCap(Paint.Cap.ROUND);
        drawPaint2 = new Paint();
        drawPaint2.setColor(paintColor2);
        drawPaint2.setAntiAlias(true);
        drawPaint2.setStrokeWidth(currentBrushSize * 2.0F);
        drawPaint2.setStyle(Paint.Style.STROKE);
        drawPaint2.setStrokeJoin(Paint.Join.ROUND);
        drawPaint2.setStrokeCap(Paint.Cap.ROUND);
        preprocessor = new PreProcessor(getContext());
        canvasPaint = new Paint(Paint.DITHER_FLAG);
    }

    public void eraseAll() {
        paths.clear();
        invalidate();
    }

    public void onClickUndo() {
        if (paths.size() > 0) {
            paths.remove(paths.size() - 1);
            invalidate();
        }

    }

    public void onClickFlip() {
        if (training) {
            training = false;
            this.setBackgroundColor(Color.parseColor("#FFFFFF"));
        } else if (!training) {
            training = true;
            this.setBackgroundColor(Color.parseColor("#1f1f2b"));
        }
    }
}
