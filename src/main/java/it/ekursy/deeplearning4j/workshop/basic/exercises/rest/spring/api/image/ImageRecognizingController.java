package it.ekursy.deeplearning4j.workshop.basic.exercises.rest.spring.api.image;

import java.io.IOException;

import org.apache.logging.log4j.Logger;

import it.ekursy.deeplearning4j.workshop.basic.exercises.rest.spring.service.ModelProcessingService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api")
class ImageRecognizingController {

    private final ModelProcessingService modelProcessingService;

    private final Logger logger;

    /**
     *
     * @param modelProcessingService
     * @param logger
     */
    ImageRecognizingController(ModelProcessingService modelProcessingService, Logger logger)
    {
        this.modelProcessingService = modelProcessingService;
        this.logger = logger;
    }

    @GetMapping("/image")
    String ping()
    {
        return "pong";
    }

    @PostMapping("/image")
    String processImage(@RequestParam MultipartFile file) throws IOException
    {
        logger.info("Processing new file");

        return modelProcessingService.processImageWithNeuralNetwork( file );
    }

}
