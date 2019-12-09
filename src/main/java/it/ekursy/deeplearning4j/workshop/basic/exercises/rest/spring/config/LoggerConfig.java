package it.ekursy.deeplearning4j.workshop.basic.exercises.rest.spring.config;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.InjectionPoint;
import org.springframework.beans.factory.config.BeanDefinition;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Scope;

@Configuration
public class LoggerConfig {

    @Bean
    @Scope(BeanDefinition.SCOPE_PROTOTYPE)
    public Logger produceLogger(InjectionPoint injectionPoint)
    {
        var classOnWired = injectionPoint.getMember().getDeclaringClass();
        return LogManager.getLogger( classOnWired );
    }
}
