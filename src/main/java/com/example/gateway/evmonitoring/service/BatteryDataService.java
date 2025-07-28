package com.example.gateway.evmonitoring.service;

import com.example.gateway.evmonitoring.model.BatteryData;
import com.example.gateway.evmonitoring.repository.BatteryDataRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.ArrayList;
import java.util.List;

@Service
public class BatteryDataService {

    @Autowired
    private BatteryDataRepository batteryDataRepository;

    public List<BatteryData> getAllData() {
        Iterable<BatteryData> iterable = batteryDataRepository.findAll();
        List<BatteryData> result = new ArrayList<>();
        iterable.forEach(result::add);
        return result;
    }

    // Option 1: Handle list results
    public BatteryData getLatestReading() {
        List<BatteryData> results = batteryDataRepository.findLatestReading();
        return results.isEmpty() ? null : results.get(0);
    }




}