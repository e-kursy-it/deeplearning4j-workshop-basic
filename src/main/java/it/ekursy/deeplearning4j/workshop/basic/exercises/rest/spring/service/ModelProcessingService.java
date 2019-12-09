package it.ekursy.deeplearning4j.workshop.basic.exercises.rest.spring.service;

import java.io.IOException;

import org.apache.logging.log4j.Logger;

import it.ekursy.deeplearning4j.workshop.basic.exercises.rest.spring.http.error.NotFoundException;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

@Component
public class ModelProcessingService {

    private final Logger logger;

    private final MultiLayerNetwork model;

    /**
     * @param logger
     * @param model
     */
    public ModelProcessingService(Logger logger, MultiLayerNetwork model)
    {
        this.logger = logger;
        this.model = model;
    }

    public String processImageWithNeuralNetwork(MultipartFile file) throws IOException
    {
        logger.info( "Processing multipart file" );

        var output = (INDArray) null;

        logger.info( "output from netowrk: {}", output );
        double max = output.getRow( 0 ).max().getDouble( 0 );
        if ( max > 0.30 ) {
            int idx = findMatchingIndex( output, max );
            return "{\"digit\":\"" + idx + "\", \"score\": " + max + "}";
        }
        else {
            throw new NotFoundException();
        }
    }

    private static int findMatchingIndex(INDArray output, double max)
    {
        throw new UnsupportedOperationException();
    }

}
