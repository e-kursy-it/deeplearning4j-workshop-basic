package it.ekursy.deeplearning4j.workshop.basic.tools;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.fetchers.SvhnDataFetcher;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.model.YOLO2;

public class ZooModelsAndTrainDataDownload {

    public static void main(String[] args) throws IOException
    {
        // dl4j's default severs are really slow, so please use mine
        DL4JResources.setBaseDownloadURL( "http://deeplearning4j.e-kursy.it/" );

        // change model download location into src/main/resources of this project ;)
        DL4JResources.setBaseDirectory( new File( "src/main/resources/" ) );

        // download pre-trained vgg16
        {
            var zooModel = VGG16.builder().numClasses( 1000 ).build();
            var computationGraph = (ComputationGraph) zooModel.initPretrained( PretrainedType.IMAGENET );
        }

        // download pre-trained yolo2
        {
            var zooModel = YOLO2.builder().numClasses( 6 ).build();
            var computationGraph = (ComputationGraph) zooModel.initPretrained();
        }

        // download pre-trained tinyYolo
        {
            var zooModel = TinyYOLO.builder().numClasses( 6 ).build();
            var computationGraph = (ComputationGraph) zooModel.initPretrained();
        }

//        {
//
//            SvhnDataFetcher fetcher = new SvhnDataFetcher();
//
//            DL4JResources.setBaseDirectory( new File( "src/main/resources/" ) );
//
//            File trainDir = fetcher.getDataSetPath( DataSetType.TRAIN );
//            File testDir = fetcher.getDataSetPath( DataSetType.TEST );
//        }

    }
}
