package tripos.partIIB.gestureflow;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Path;
import android.graphics.Rect;
import android.os.Build;
import android.util.Log;
import android.util.Pair;
import android.widget.Toast;

import androidx.annotation.RequiresApi;

import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.checkerframework.checker.units.qual.A;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.sql.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import tripos.partIIB.gestureflow.Point;
import tripos.partIIB.gestureflow.Rectangle;
import tripos.partIIB.gestureflow.ml.Modelpredict;
//import tripos.partIIB.gestureflow.ml.Modelgen2;

public class PreProcessor {
    Interpreter tflite;
    Interpreter tflitepred;
    Context ctx;

    private static float[] path2bound =
            {2.208985677220568F,    // arrow
            1.6108276210417536F,    // caret
            1.289421818299565F,     // check
            2.1957364755003153F,    // circle
            2.4805302616619085F,    // delete mark
            1.6536689897070318F,    // left curly brace
            1.6483579646649897F,    // left sq bracket
            2.0913029646636727F,    // pigtail
            1.634860296757598F,     // question mark
            2.540117628462438F,     // rectangle
            1.7119027885956875F,    // right curly brace
            1.7223687479949679F,    // right sq bracket
            3.632571077476236F,     // star
            2.2030158296540674F,    // triangle
            1.5882281548389237F,    // v
            2.595416053776574F};    // x

    private static float[] x2y =
            {1.6320157158367543F,    // arrow
            0.933919749979444F,     // caret
            1.0343318344180665F,    // check
            0.9314208236789377F,    // circle
            0.8888682345943351F,    // delete mark
            0.3871966776700537F,    // left curly brace
            0.5509602774138739F,    // left sq bracket
            1.3918053411552935F,    // pigtail
            0.5563573625459236F,    // question mark
            1.2661251476849458F,    // rectangle
            0.41349352375642284F,   // right curly brace
            0.6519216687516202F,    // right sq bracket
            1.050711595699299F,     // star
            1.3086574009828558F,    // triangle
            0.9153801687059542F,    // v
            0.7761957495988906F};   // x

    private static float[] cx =
            {0.6370919875783057F,   // arrow
            0.498602068397655F,     // caret
            0.4503913039353075F,    // check
            -0.08623731426241352F,  // circle
            0.32427801723143285F,   // delete mark
            -0.3969159175529128F,   // left curly brace
            -0.5026438988192298F,   // left sq bracket
            0.5138770272466476F,    // pigtail
            0.47126791201816537F,   // question mark
            0.43222554088458937F,   // rectangle
            0.4027995579711682F,    // right curly brace
            0.5217582116386501F,    // right sq bracket
            0.33949921346511075F,   // star
            -0.05884566514457435F,  // triangle
            0.4663827860294973F,    // v
            0.5294531514654011F};   // x

    private static float[] cy =
            {-0.5624576724252005F,   // arrow
            -0.44909286988413116F,     // caret
            -0.052845985972343766F,    // check
            0.41245738550360633F,  // circle
            0.5496579000889739F,   // delete mark
            0.4625738194083566F,   // left curly brace
            0.47433600079076377F,   // left sq bracket
            -0.341429437166188F,    // pigtail
            0.027915779867800382F,   // question mark
            0.4343112219752317F,   // rectangle
            0.4420422615300465F,    // right curly brace
            0.421953903397359F,    // right sq bracket
            -0.4203330847076779F,   // star
            0.6246256069757528F,  // triangle
            0.3831092934694503F,    // v
            0.3539942137718633F};   // x

    private static Map<String, Integer> gest2int;

    static {
        gest2int = new HashMap<String, Integer>();
        gest2int.put("arrow", 0);
        gest2int.put("caret", 1);
        gest2int.put("check", 2);
        gest2int.put("circle", 3);
        gest2int.put("delete mark", 4);
        gest2int.put("left curly brace", 5);
        gest2int.put("left sq bracket", 6);
        gest2int.put("pigtail", 7);
        gest2int.put("question mark", 8);
        gest2int.put("rectangle", 9);
        gest2int.put("right curly brace", 10);
        gest2int.put("right sq brace", 11);
        gest2int.put("star", 12);
        gest2int.put("triangle", 13);
        gest2int.put("v", 14);
        gest2int.put("x", 15);
    }

    private static String[] int2gest =
            {"arrow",
            "caret",
            "check",
            "circle",
            "delete mark",
            "left curly brace",
            "left sq bracket",
            "pigtail",
            "question mark",
            "rectangle",
            "right curly brace",
            "right sq brace",
            "star",
            "triangle",
            "v",
            "x"};

    PreProcessor(Context _ctx){
        ctx = _ctx;
        try {
            Interpreter.Options tfoptions = new Interpreter.Options();
            tfoptions.setNumThreads(16);
            tflite = new Interpreter(loadModelFile(ctx), tfoptions);
            tflitepred = new Interpreter(loadModelFilePred(ctx), tfoptions);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public float path_length(ArrayList<Point> points) {
        float path_length = 0.0F;
        for (int i = 1; i < points.size(); i++) {
            path_length += point_distance(points.get(i), points.get(i - 1));
        }
        return path_length;
    }

    public float point_distance(Point A, Point B) {
        float distance = 0.0F;
        float x = A.getX() - B.getX();
        float y = A.getY() - B.getY();
        distance += Math.pow(Math.pow(x, 2) + Math.pow(y, 2), 0.5);
        return distance;
    }

    public ArrayList<Point> resample(ArrayList<Point> points, int sample_num) {
        ArrayList<Point> resampled_points = new ArrayList<Point>();
        float avg_dist = path_length(points) / (sample_num - 1);
        float run_dist = 0.0F;
        resampled_points.add(points.get(0));

        int i = 1;
        while (resampled_points.size() < sample_num - 1) {
            float dist = point_distance(points.get(i), points.get(i - 1));

            if ((run_dist + dist) >= avg_dist) {
                float new_x = points.get(i - 1).getX() + ((avg_dist - run_dist) / dist) * (points.get(i).getX() - points.get(i - 1).getX());
                float new_y = points.get(i - 1).getY() + ((avg_dist - run_dist) / dist) * (points.get(i).getY() - points.get(i - 1).getY());
                Point resampled_point = new Point(new_x, new_y);
                resampled_points.add(resampled_point);
                points.add(i, resampled_point);
                run_dist = 0;
            } else {
                run_dist += dist;
            }
            i += 1;
        }
        return resampled_points;
    }

    public Point centroid(ArrayList<Point> points) {
        float x_sum = 0.0F;
        float y_sum = 0.0F;
        int n = points.size();

        for (Point point : points) {
            x_sum += point.getX();
            y_sum += point.getY();
        }

        Point centroid = new Point((x_sum / n), (y_sum / n));
        return centroid;
    }

    public float indicative_angle(ArrayList<Point> points) {
        float w = 0.0F;
        Point c = centroid(points);
        w += Math.atan((c.getY() - points.get(0).getY()) / (c.getX() - points.get(0).getX()));
        return w;
    }

    public ArrayList<Point> rotate(ArrayList<Point> points) {
        Point c = centroid(points);
        float w = indicative_angle(points);
        ArrayList<Point> rotated_points = new ArrayList<Point>();

        for (Point point : points) {
            float rot_x = 0.0F;
            float rot_y = 0.0F;
            rot_x += (point.getX() - c.getX()) * Math.cos(w) - (point.getY() - c.getY()) * Math.sin(w) + c.getX();
            rot_y += (point.getX() - c.getX()) * Math.sin(w) + (point.getY() - c.getY()) * Math.cos(w) + c.getY();
            Point new_point = new Point(rot_x, rot_y);
            rotated_points.add(new_point);
        }

        return rotated_points;
    }

    public Rectangle bounding_box(ArrayList<Point> points) {

        float min_x = points.get(0).getX();
        float min_y = points.get(0).getY();
        float max_x = points.get(0).getX();
        float max_y = points.get(0).getY();

        for (Point point : points) {
            if (point.getX() < min_x){
                min_x = point.getX();
            } else if (point.getX() > max_x){
                max_x = point.getX();
            }

            if (point.getY() < min_y){
                min_y = point.getY();
            } else if (point.getY() > max_y){
                max_y = point.getY();
            }
        }
        Rectangle box = new Rectangle(min_x, min_y, max_x, max_y);

        return (box);
    }

    float[] Gest2Vec(ArrayList<Point> gesture){
        float[] vec = new float[gesture.size() * 2];
        for(int i = 0; i < gesture.size(); i++){
            vec[i * 2] = gesture.get(i).getX();
            vec[(i * 2) + 1] = gesture.get(i).getY();
        }
        return vec;
    }

    float[][][] Gest2Inp(ArrayList<Point> gesture, int type){
        float[][][] vec = new float[1][30][18];
        for(int i = 0; i < gesture.size(); i++){
            vec[0][i][0] = gesture.get(i).getX();
            vec[0][i][1] = gesture.get(i).getY();
            for(int p = 2; p < 18; p++) {
                if (p - 2 == type){
                    vec[0][i][p] = 1.0F;
                } else {
                    vec[0][i][p] = 0.0F;
                }
            }
        }
        return vec;
    }

    float[][][] Gest2Inp2(ArrayList<Point> gesture){
        float[][][] vec = new float[1][30][2];
        for(int i = 0; i < gesture.size(); i++){
            vec[0][i][0] = gesture.get(i).getX();
            vec[0][i][1] = gesture.get(i).getY();
        }
        return vec;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    ArrayList<Integer> predict(float[] raw){
        ArrayList<Pair<Float, Integer>> Raw = new ArrayList<>();
        for (int i = 0; i < raw.length; i++){
            Pair<Float, Integer> newPair = new Pair<>((Float) raw[i], i);
            Raw.add(newPair);
        }

        ArrayList<Integer> prediction = new ArrayList<Integer>();
        Collections.sort(Raw, Comparator.comparingDouble(p -> -p.first.doubleValue()));

        int j = 0;
        for (int i = 0; i < raw.length; i++){
            if (Float.compare(Raw.get(i).first, Raw.get(15).first * 0.1F) >= 0){
                prediction.add(Raw.get(i).second);
            }
        }

        ArrayList<String> answers = new ArrayList<>();
        for (Integer i : prediction){
            answers.add(int2gest[i]);
        }

        Log.d("MyTag", "model_predict: " + Arrays.toString(answers.toArray()));
        return prediction;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    float[] model_predict(ArrayList<Point> gesture){
        float[][][] input = Gest2Inp2(gesture);
        float[][][] output = new float[1][30][16];
        tflitepred.run(input, output);
        return output[0][gesture.size()];
    }

//    @RequiresApi(api = Build.VERSION_CODES.N)
//    float[] model_predict (ArrayList<Point> gesture){
//        try {
//            Modelpredict model = Modelpredict.newInstance(ctx);
//
//            // Creates inputs for reference.
//            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 1, 2}, DataType.FLOAT32);
//
//            float[] vec = Gest2Vec(gesture);
//            TensorBuffer outputFeature0 = null;
//            ByteBuffer byteBuffer = ByteBuffer.allocate(8);
//
//            for(int i = 0; i < vec.length; i+=2){
//                byteBuffer.clear();
//                byteBuffer.putFloat(0, vec[i]);
//                byteBuffer.putFloat(4, vec[i + 1]);
//                inputFeature0.loadBuffer(byteBuffer);
//                Modelpredict.Outputs outputs = model.process(inputFeature0);
//                outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
//            }
//
//            float[] output = new float[16];
//            for(int i = 0; i < 64; i+=4) {
//                output[i/4] = outputFeature0.getBuffer().getFloat(i);
//            }
//
//            // Releases model resources if no longer used.
//            model.close();
//            return output;
//        } catch (IOException e) {
//            // TODO Handle the exception
//            return null;
//        }
//    }

    private MappedByteBuffer loadModelFile(Context ctx) throws IOException{
        AssetFileDescriptor fileDescriptor =  ctx.getAssets().openFd("modelgenerate.tflite");
        FileInputStream fileInputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffSets = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffSets, declaredLength);
    }

    private MappedByteBuffer loadModelFilePred(Context ctx) throws IOException{
        AssetFileDescriptor fileDescriptor =  ctx.getAssets().openFd("modelpredict.tflite");
        FileInputStream fileInputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffSets = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffSets, declaredLength);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    ArrayList<Point> model_generate(ArrayList<Point> gesture, Context ctx, int type, Point start){
        float[][][] input = Gest2Inp(gesture, type);
        float[][][] output = new float[1][30][120];
        tflite.run(input, output);

        for (int i = gesture.size(); i < Math.min(gesture.size() + 10, 29); i++) {
            float[] next = forecast(output[0][i], type);
            input[0][i + 1][0] = next[0];
            input[0][i + 1][1] = next[1];
            for (int p = 2; p < 18; p++) {;
                if (p - 2 == type) {
                    input[0][i + 1][p] = 1.0F;
                } else {
                    input[0][i + 1][p] = 0.0F;
                }
            }
            tflite.run(input, output);
        }

        ArrayList<Point> result = new ArrayList<Point>();
        for(int i = 0; i < 30; i++){
            Point newpoint = new Point(input[0][i][0], input[0][i][1]);
            result.add(newpoint);
        }

        return result;
    }

    public ArrayList<Point> reconstruct(ArrayList<Point> offs, Point start){
        ArrayList<Point> output = new ArrayList<Point>();
        output.add(start);

        for(Point offset : offs){
            output.add(new Point(output.get(output.size() - 1).getX() + (offset.getX() * 5.950835696596043F), output.get(output.size() - 1).getY() + (offset.getY() * 5.950835696596043F)));
        }
        return output;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public float[] forecast(float[] previous, int type){
        float[] pi = Arrays.copyOfRange(previous, 0, 20);
        float[] mu1 = Arrays.copyOfRange(previous, 20, 40);
        float[] mu2 = Arrays.copyOfRange(previous, 40, 60);
        float[] std1 = Arrays.copyOfRange(previous, 60, 80);
        float[] std2 = Arrays.copyOfRange(previous, 80, 100);
        float[] rho = Arrays.copyOfRange(previous, 100, 120);
        int[] enumerate = IntStream.range(0, 20).toArray();

        double[] pi_doub = IntStream.range(0, pi.length).mapToDouble(i -> pi[i]).toArray();
        EnumeratedIntegerDistribution mul = new EnumeratedIntegerDistribution(enumerate, pi_doub);
        int d = mul.sample();

        int temperature = 300;
        double[] means = {(double) mu1[d], (double) mu2[d]};
 //       double[][] covariances = {{(double) Math.abs(Math.pow(std1[d], 2)) / temperature, (double) Math.abs(Math.pow(rho[d], 2)) / temperature}, {(double) Math.abs(Math.pow(rho[d], 2)) / temperature, (double) Math.abs(Math.pow(std2[d], 2)) / temperature}};
        double[][] covariances = {{Math.abs(std1[d] / 300), 0}, {0, Math.abs(std2[d]/300)}};

        MultivariateNormalDistribution mnd = new MultivariateNormalDistribution(means, covariances);

        double[] temp = mnd.sample();
  //      Log.d("myTag", Arrays.toString(means));
        float[] next = {(float) temp[0], (float) temp[1]};

        return next;
    }

    public ArrayList<Point> resize(ArrayList<Point> points, float sizeX, float sizeY, Rectangle bound) {
        float[] b_box = {bound.getWidth(), bound.getHeight()};

        ArrayList<Point> resized_points = new ArrayList<Point>();

        for (Point point : points) {
            float new_x = point.getX() * sizeX / b_box[0];
            float new_y = point.getY() * sizeY / b_box[1];
            Point new_point = new Point(new_x, new_y);
            resized_points.add(new_point);
        }

        return resized_points;
    }

    public ArrayList<Point> normalize(ArrayList<Point> points, Rectangle bound){
        Point k = new Point(0F, 0F);
        float size = 200;

        ArrayList<Point> resized_points = resize(points, size, size, bound);
        ArrayList<Point> translated_points = translate(resized_points, k);

        return translated_points;
    }

    public Path draw(ArrayList<Point> gesture){
        Path drawPath = new Path();
        drawPath.moveTo(gesture.get(0).getX(), gesture.get(0).getY());
        for(int i=1; i<gesture.size(); i++){
            float mX = gesture.get(i-1).getX();
            float mY = gesture.get(i-1).getY();
            float X = gesture.get(i).getX();
            float Y = gesture.get(i).getY();
            drawPath.quadTo(mX, mY, (X + mX) / 2, (Y + mY) / 2);
        }
        return drawPath;
    }

    public ArrayList<Point> offsets(ArrayList<Point> points){
        ArrayList<Point> offsets = new ArrayList<Point>();

        for(int i = 1; i < points.size(); i++){
            Point offset = new Point((points.get(i).getX() - points.get(i - 1).getX()) / 5.950835696596043F, (points.get(i).getY() - points.get(i - 1).getY()) / 5.950835696596043F);
            offsets.add(offset);
        }

        return offsets;
    }

    public ArrayList<Point> translate(ArrayList<Point> points, Point offset) {
        ArrayList<Point> translated_points = new ArrayList<Point>();

        for (Point point: points) {
            float new_x = point.getX() + offset.getX() - centroid(points).getX();
            float new_y = point.getY() + offset.getY() - centroid(points).getY();
            Point new_point = new Point(new_x, new_y);
            translated_points.add(new_point);
        }

        return translated_points;
    }

    public ArrayList<Point> overlay(ArrayList<Point> points, Point offset) {
        ArrayList<Point> translated_points = new ArrayList<Point>();

        for (Point point: points) {
            float new_x = point.getX() + offset.getX() - points.get(0).getX();
            float new_y = point.getY() + offset.getY() - points.get(0).getY();
            Point new_point = new Point(new_x, new_y);
            translated_points.add(new_point);
        }

        return translated_points;
    }

    public Rectangle dynamic_bounding_box(ArrayList<Point> gesture, int width, int height, int type){
        int wall_right = (int) (0.9 * width);
        int wall_left = (int) (0.1 * width);
        int wall_top = (int) (0.1 * height);
        int wall_bot = (int) (0.9 * height);
        int box_width;
        int box_height;
        Point start = gesture.get(0);

        if (cx[type] > 0) {
            box_width = (int) Math.min(((wall_right - start.getX()) / (Math.abs(cx[type]) + 0.5)), (0.4 * width));
            box_height = (int) (box_width / x2y[type]);
        } else {
            box_width = (int) Math.min(((start.getX() - wall_left) / (Math.abs(cx[type]) + 0.5)), (0.4 * width));
            box_height = (int) (box_width / x2y[type]);
        }

        Point centroid_est = new Point(start.getX() + (box_width * cx[type]), start.getY() + (box_height * cy[type]));
        Rectangle true_bound = bounding_box(gesture);
        float min_x = Math.min(Math.max(wall_left, centroid_est.getX() - (box_width / 2)), true_bound.bot_left.getX());
        float min_y = Math.min(Math.max(wall_top, centroid_est.getY() - (box_height / 2)), true_bound.top_right.getY());
        float max_x = Math.max(Math.min(wall_right, centroid_est.getX() + (box_width / 2)), true_bound.top_right.getY());
        float max_y = Math.max(Math.min(wall_bot, centroid_est.getY() + (box_height / 2)), true_bound.bot_left.getY());

        return new Rectangle(min_x, min_y, max_x, max_y);
    }

    public ArrayList<Point> Approx2Gest(float[] approx){
        ArrayList<Point> gesture = new ArrayList<Point>();
        for(int i = 0; i < approx.length; i+=3){
            float x = approx[i + 1];
            float y = approx[i + 2];
            Point point = new Point(x, y);
            gesture.add(point);
        }
        return gesture;
    }

    public String Int2Gest(Integer i){
        return int2gest[i];
    }

    public Integer Gest2Int(String gest){
        return gest2int.get(gest);
    }

    public Float Path2Bound(Integer i){
        return path2bound[i];
    }

    public Float BX2Y(Integer i){
        return x2y[i];
    }
}
