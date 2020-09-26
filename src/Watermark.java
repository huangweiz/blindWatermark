import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;


public class Watermark {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static List<Mat> planes = new ArrayList<Mat>();
    private static List<Mat> allPlanes = new ArrayList<Mat>();

    public static void main(String[] args) {
        Mat src = Imgcodecs.imread("./images/source.jpg");
        Mat watermark = Imgcodecs.imread("./images/aaa.png");
        new ShowImage(src);

//        Mat dst = addWatermarkWithText(src, "huang");
        Mat dst = addWatermarkWithImage(src, watermark);

        new ShowImage(dst);
        Imgcodecs.imwrite("./images/out.jpg", dst);
        Mat cod = getWatermark(dst);

        new ShowImage(cod);
        Imgcodecs.imwrite("./images/cod.jpg", cod);


    }


    /*
        添加文本水印，在单通道进行添加
     */
    public static Mat addWatermarkWithText(Mat image, String text) {
        Mat complexImage = new Mat();
        Mat padded = splitSrc(image);

//        new ShowImage(padded);
        padded.convertTo(padded, CvType.CV_32F);    // 数据类型转换
        planes.add(padded);
        planes.add(Mat.zeros(padded.size(), CvType.CV_32F));

        // 用于合并通道，即将planes中的两个单通道图合并为一个
        Core.merge(planes, complexImage);

//        System.out.println(complexImage.channels());

        Core.dft(complexImage, complexImage);

        // 添加文本水印
        Scalar scalar = new Scalar(0, 0, 0);
        Point point = new Point(40, 40);

        Imgproc.putText(complexImage, text, point, Core.FONT_HERSHEY_DUPLEX, 1D, scalar);

        // 图像旋转，0代表沿x轴旋转，任意整数代表沿y轴旋转，任意负数代表沿x,y轴同时旋转
        Core.flip(complexImage, complexImage, -1);

        // 对称加水印
        Imgproc.putText(complexImage, text, point, Core.FONT_HERSHEY_DUPLEX, 1D, scalar);
        Core.flip(complexImage, complexImage, -1);


        return decode(complexImage, allPlanes);
    }


    public static Mat addWatermarkWithImage(Mat image, Mat watermark) {
        Mat complexImage = new Mat();
        Mat padded = splitSrc(image);

//        new ShowImage(padded);
        padded.convertTo(padded, CvType.CV_32F);    // 数据类型转换
        planes.add(padded);
        planes.add(Mat.zeros(padded.size(), CvType.CV_32F));

        // 用于合并通道，即将planes中的两个单通道图合并为一个
        Core.merge(planes, complexImage);

//        System.out.println(complexImage.channels());

        Core.dft(complexImage, complexImage);
        Mat imageROI;
        imageROI = new Mat(complexImage, new Rect(0, 0, watermark.cols(), watermark.rows()));
        watermark.copyTo(imageROI);
        Core.flip(complexImage, complexImage, -1);

        imageROI = new Mat(complexImage, new Rect(0, 0, watermark.cols(), watermark.rows()));
        watermark.copyTo(imageROI);
        Core.flip(complexImage, complexImage, -1);

        new ShowImage(image);
        return decode(complexImage, allPlanes);

    }


    /*
        将加了水印的频谱图进行解码，解码一个单通道的，再和为加水印的其他通道的图进行合并
     */
    public static Mat decode(Mat complexImage, List<Mat> allPlanes) {
        Mat invDFT = new Mat();
        Core.idft(complexImage, invDFT, Core.DFT_SCALE | Core.DFT_REAL_OUTPUT, 0);
        Mat restoredImage = new Mat();
        invDFT.convertTo(restoredImage, CvType.CV_8U);

        if (allPlanes.size() == 0) {
            allPlanes.add(restoredImage);
        } else {
            allPlanes.set(0, restoredImage);
        }

        Mat lastImage = new Mat();
        Core.merge(allPlanes, lastImage);
        return lastImage;
    }


    public static Mat getWatermark(Mat image) {
        List<Mat> planes = new ArrayList<>();
        Mat complexImage = new Mat();
        Mat padded = splitSrc(image);
        padded.convertTo(padded, CvType.CV_32F);

        planes.add(padded);
        planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
        Core.merge(planes, complexImage);

        Core.dft(complexImage, complexImage);

        Mat mag = createMag(complexImage);

        planes.clear();

        return mag;


    }

    private static Mat createMag(Mat complexImage) {
        List<Mat> newPlanes = new ArrayList<>();
        Mat mag = new Mat();
        Core.split(complexImage, newPlanes);

        // 计算二维矢量的幅值
        Core.magnitude(newPlanes.get(0), newPlanes.get(1), mag);

        Core.add(Mat.ones(mag.size(), CvType.CV_32F), mag, mag);
        Core.log(mag, mag);
        shiftDFT(mag);
        mag.convertTo(mag, CvType.CV_8UC1);
        Core.normalize(mag, mag, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);

        return mag;
    }

    private static void shiftDFT(Mat image) {
        image = image.submat(new Rect(0, 0, image.cols() & -2, image.rows() & -2));
        int cx = image.cols() / 2;
        int cy = image.rows() / 2;

        Mat q0 = new Mat(image, new Rect(0, 0, cx, cy));
        Mat q1 = new Mat(image, new Rect(cx, 0, cx, cy));
        Mat q2 = new Mat(image, new Rect(0, cy, cx, cy));
        Mat q3 = new Mat(image, new Rect(cx, cy, cx, cy));
        Mat tmp = new Mat();
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

    }


    /*
        返回一个单通道的图
     */
    public static Mat splitSrc(Mat mat) {
        mat = optimizeImageDim(mat);

        // 用于通道分离
        Core.split(mat, allPlanes);

        Mat padded = new Mat();

        if (allPlanes.size() > 1) {
            for (int i = 0; i < allPlanes.size(); i++) {
                if (i == 0) {
                    padded = allPlanes.get(i);
                }

            }
        } else {
            padded = mat;
        }


        return padded;

    }


    /*
        为了加快 傅里叶变换的速度，对要处理的图片尺寸进行优化
     */
    private static Mat optimizeImageDim(Mat image) {
        Mat dst = new Mat();

        // 得到dft算法最合适的行列数
        // 函数返回给定向量尺寸的傅里叶最优尺寸大小
        int rows = Core.getOptimalDFTSize(image.rows());
        int cols = Core.getOptimalDFTSize(image.cols());

        // 边界填充
        // 四个参数分别为向上、下、左、右填充边缘的大小
        // 填充类型为常量
        Core.copyMakeBorder(image, dst, 0, rows - image.rows(), 0, cols - image.cols(), Core.BORDER_CONSTANT, Scalar.all(0));

//        new ShowImage(dst);

        return dst;

    }


}
