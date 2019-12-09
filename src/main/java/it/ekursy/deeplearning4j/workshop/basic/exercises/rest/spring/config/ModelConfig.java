package it.ekursy.deeplearning4j.workshop.basic.exercises.rest.spring.config;

import java.io.File;
import java.io.IOException;

import org.apache.logging.log4j.Logger;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.nn.graph.ComputationGraph;
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
    public ComputationGraph model() throws IOException
    {
        // dl4j's default severs are really slow, so please use mine
        DL4JResources.setBaseDownloadURL( "http://deeplearning4j.e-kursy.it/" );

        // change model download location into src/main/resources of this project ;)
        DL4JResources.setBaseDirectory( new File( "src/main/resources/" ) );

        var model = (ComputationGraph) null;
        logger.info( "Creating new model!" );
        return model;
    }

}
