package com.example.gateway.evmonitoring.config;
import com.azure.cosmos.CosmosClientBuilder;
import com.azure.cosmos.DirectConnectionConfig;
import com.azure.cosmos.GatewayConnectionConfig;
import com.azure.spring.data.cosmos.config.AbstractCosmosConfiguration;
import com.azure.spring.data.cosmos.repository.config.EnableCosmosRepositories;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@EnableCosmosRepositories(basePackages = "com.example.gateway.evmonitoring.repository")
public class CosmosConfiguration extends AbstractCosmosConfiguration {

    @Value("${azure.cosmos.uri}")
    private String uri;
    @Value("${azure.cosmos.key}")
    private String key;
    @Value("${azure.cosmos.database}")
    private String dbName;

    @Bean
    public CosmosClientBuilder cosmosClientBuilder() {
        return new CosmosClientBuilder()
                .endpoint(uri)
                .key(key)
                .contentResponseOnWriteEnabled(true)
                .gatewayMode();
    }
    @Override
    protected String getDatabaseName() {
        return dbName;
    }
}