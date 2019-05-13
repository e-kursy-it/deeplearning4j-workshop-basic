package it.ekursy.deeplearning4j.workshop.basic.exercises;

import static org.bytedeco.javacpp.opencv_core.CV_8U;
import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.Point;
import static org.bytedeco.javacpp.opencv_core.Scalar;
import static org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

import java.io.File;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import it.ekursy.deeplearning4j.workshop.basic.exercises.yolo2custom.Yolo2CustomDataProvider;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

public class Yolo2CustomTrainer {

    private static final Logger log = LogManager.getLogger();

    static int seed = 123;

    static int nClasses = 10;

    // parameters for the Yolo2OutputLayer
    static int nBoxes = 5;
    static double lambdaNoObj = 0.5;
    static double lambdaCoord = 1.0;

    static double learningRate = 1e-4;
    static double lrMomentum = 0.9;

    public static void main(String[] args) throws java.lang.Exception
    {
        // dl4j's default severs are really slow, so please use mine
        DL4JResources.setBaseDownloadURL( "http://deeplearning4j.e-kursy.it/" );

        // change model download location into src/main/resources of this project ;)
        DL4JResources.setBaseDirectory( new File( "src/main/resources/" ) );

        // parameters matching the pretrained TinyYOLO model
        int width = 416;
        int height = 416;
        int nChannels = 3;
        int gridWidth = 13;
        int gridHeight = 13;

        double[][] priorBoxes = { { 2, 5 }, { 2.5, 6 }, { 3, 7 }, { 3.5, 8 }, { 4, 9 } };
        double detectionThreshold = 0.5;

        // parameters for the training phase
        int batchSize = 10;
        int nEpochs = 20;

        Random rng = new Random( seed );

        var trainDir = Paths.get( "src/main/resources/data/digitalclock/frames_every_8th" );
        var testDir = Paths.get( "src/main/resources/data/digitalclock/frames_every_8th" );

        FileSplit trainData = new FileSplit( trainDir.toFile(), NativeImageLoader.ALLOWED_FORMATS, rng );
        FileSplit testData = new FileSplit( testDir.toFile(), NativeImageLoader.ALLOWED_FORMATS, rng );

        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader( height, width, nChannels, gridHeight, gridWidth,
                new Yolo2CustomDataProvider( trainDir ) );
        recordReaderTrain.initialize( trainData );

        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader( height, width, nChannels, gridHeight, gridWidth,
                new Yolo2CustomDataProvider( testDir ) );
        recordReaderTest.initialize( testData );

        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator( recordReaderTrain, batchSize, 1, 1, true );
        train.setPreProcessor( new ImagePreProcessingScaler( 0, 1 ) );

        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator( recordReaderTest, 1, 1, 1, true );
        test.setPreProcessor( new ImagePreProcessingScaler( 0, 1 ) );

        ComputationGraph model;
        String modelFilename = "model.zip";

        if ( new File( modelFilename ).exists() ) {
            log.info( "Load model..." );

            model = ModelSerializer.restoreComputationGraph( modelFilename );
        }
        else {
            log.info( "Build model..." );

            ComputationGraph pretrained = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
            INDArray priors = Nd4j.create( priorBoxes );

            model = initCustomModel( pretrained, priors );

            log.info( "Train model..." );

            model.setListeners( new ScoreIterationListener( 1 ) );
            for (int i = 0; i < nEpochs; i++) {
                train.reset();
                while ( train.hasNext() ) {
                    model.fit( train.next() );
                }
                log.info( "*** Completed epoch {} ***", i );
            }
            ModelSerializer.writeModel( model, modelFilename, true );
        }

        // visualize results on the test set
        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame frame = new CanvasFrame( "YoloCustomTrainer" );
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer( 0 );
        List<String> labels = train.getLabels();
        test.setCollectMetaData( true );
        while ( test.hasNext() && frame.isVisible() ) {
            org.nd4j.linalg.dataset.DataSet ds = test.next();
            RecordMetaDataImageURI metadata = (RecordMetaDataImageURI) ds.getExampleMetaData().get( 0 );
            INDArray features = ds.getFeatures();
            INDArray results = model.outputSingle( features );
            List<DetectedObject> objs = yout.getPredictedObjects( results, detectionThreshold );
            File file = new File( metadata.getURI() );
            log.info( file.getName() + ": " + objs );

            Mat mat = imageLoader.asMat( features );
            Mat convertedMat = new Mat();
            mat.convertTo( convertedMat, CV_8U, 255, 0 );
            int w = metadata.getOrigW() * 2;
            int h = metadata.getOrigH() * 2;
            Mat image = new Mat();
            resize( convertedMat, image, new Size( w, h ) );
            for (DetectedObject obj : objs) {
                double[] xy1 = obj.getTopLeftXY();
                double[] xy2 = obj.getBottomRightXY();
                String label = labels.get( obj.getPredictedClass() );
                int x1 = (int) Math.round( w * xy1[ 0 ] / gridWidth );
                int y1 = (int) Math.round( h * xy1[ 1 ] / gridHeight );
                int x2 = (int) Math.round( w * xy2[ 0 ] / gridWidth );
                int y2 = (int) Math.round( h * xy2[ 1 ] / gridHeight );
                rectangle( image, new Point( x1, y1 ), new Point( x2, y2 ), Scalar.RED );
                putText( image, label, new Point( x1 + 2, y2 - 2 ), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN );
            }
            frame.setTitle( new File( metadata.getURI() ).getName() + " - HouseNumberDetection" );
            frame.setCanvasSize( w, h );
            frame.showImage( converter.convert( image ) );
            frame.waitKey();
        }
        frame.dispose();
    }

    private static ComputationGraph initCustomModel(ComputationGraph pretrained, INDArray priors)
    {
        // extending model howto
        //
        // https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-zoo/src/main/java/org/deeplearning4j/zoo/model/TinyYOLO.java#L57
        //
        return null;
    }
}
