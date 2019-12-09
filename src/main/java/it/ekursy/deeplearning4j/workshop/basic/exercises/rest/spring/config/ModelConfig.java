package it.ekursy.deeplearning4j.workshop.basic.exercises.rest.spring.config;

import java.io.File;
import java.io.IOException;

import org.apache.logging.log4j.Logger;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class ModelConfig {

    private final Logger logger;

    /**
     *
     * @param logger
     */
    public ModelConfig(Logger logger)
    {
        this.logger = logger;
    }

    @Bean
    public MultiLayerNetwork model() throws IOException
    {
        var model = (MultiLayerNetwork) null;
        logger.info( "Creating new model!" );
        return model;
    }

}
