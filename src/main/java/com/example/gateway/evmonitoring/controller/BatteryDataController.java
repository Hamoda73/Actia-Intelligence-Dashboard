package com.example.gateway.evmonitoring.controller;
import com.example.gateway.evmonitoring.model.BatteryData;
import com.example.gateway.evmonitoring.repository.BatteryDataRepository;
import com.example.gateway.evmonitoring.service.BatteryDataService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.time.Instant;
import java.time.ZonedDateTime;
import java.util.List;

@RestController
@RequestMapping("/api/battery")
public class BatteryDataController {

    @Autowired
    private BatteryDataService batteryDataService;


    @GetMapping
    public ResponseEntity<List<BatteryData>> getAllBatteryData() {
        return ResponseEntity.ok(batteryDataService.getAllData());
    }

    @GetMapping("/latest")
    public ResponseEntity<BatteryData> getLatest() {
        BatteryData data = batteryDataService.getLatestReading();

        if (data == null) {
            return ResponseEntity.notFound().build();
        }

        long lastModified = data.getSystemProperties() != null
                ? Instant.parse(data.getSystemProperties().getIothubEnqueuedTime())
                .toEpochMilli()
                : System.currentTimeMillis();

        return ResponseEntity.ok()
                .lastModified(lastModified)
                .body(data);
    }
}