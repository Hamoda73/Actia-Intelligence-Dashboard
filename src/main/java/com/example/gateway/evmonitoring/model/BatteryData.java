package com.example.gateway.evmonitoring.model;

import com.azure.spring.data.cosmos.core.mapping.Container;
import com.azure.spring.data.cosmos.core.mapping.PartitionKey;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.springframework.data.annotation.Id;

import java.time.Instant;

@Data
@Container(containerName = "telemetry")
public class BatteryData {

    @Id
    private String id;
    @PartitionKey
    private String evdata;
    private SystemProperties systemProperties;
    private String iothubName;
    @JsonProperty("Body")
    private Body body = new Body();


    @Data
    public static class SystemProperties {

        public long getEnqueuedTimeMillis() {
            if (this.iothubEnqueuedTime == null) {
                return 0L;
            }
            try {
                return Instant.parse(iothubEnqueuedTime).toEpochMilli();
            } catch (Exception e) {
                return 0L;
            }
        }

        @JsonProperty("iothub-connection-device-id")
        private String iothubConnectionDeviceId;

        @JsonProperty("iothub-connection-auth-method")
        private String iothubConnectionAuthMethod;

        @JsonProperty("iothub-connection-auth-generation-id")
        private String iothubConnectionAuthGenerationId;

        @JsonProperty("iothub-content-type")
        private String iothubContentType;

        @JsonProperty("iothub-content-encoding")
        private String iothubContentEncoding;

        @JsonProperty("iothub-enqueuedtime")
        private String iothubEnqueuedTime;

        @JsonProperty("iothub-message-source")
        private String iothubMessageSource;
    }

    @Data
    public static class Body {
        private String timestamp;
        private String deviceId;
        private String sensorId;
        private String type;
        private String vehicleState;
        private String chargeState;
        private double batteryVoltage;
        private double motorCurrent;
        private double stateOfCharge;
        private double maxCellVoltage;
        private double minCellVoltage;
        private double maxTemperature;
        private double minTemperature;
        private Metadata metadata;
    }

    @Data
    public static class Metadata {
        private String source;
        private int row_index;
    }
}