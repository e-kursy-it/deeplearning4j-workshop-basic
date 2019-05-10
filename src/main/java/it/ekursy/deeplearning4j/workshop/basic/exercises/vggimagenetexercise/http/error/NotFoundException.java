package it.ekursy.deeplearning4j.workshop.basic.exercises.vggimagenetexercise.http.error;

import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

public class NotFoundException extends ResponseStatusException {

    public NotFoundException()
    {
        super( HttpStatus.NOT_FOUND );
    }
}
