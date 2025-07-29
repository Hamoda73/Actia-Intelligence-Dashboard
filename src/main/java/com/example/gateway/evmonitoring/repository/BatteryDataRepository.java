package com.example.gateway.evmonitoring.repository;
import com.azure.cosmos.implementation.guava25.base.Optional;
import com.azure.spring.data.cosmos.repository.CosmosRepository;
import com.azure.spring.data.cosmos.repository.Query;
import com.example.gateway.evmonitoring.model.BatteryData;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface BatteryDataRepository extends CosmosRepository<BatteryData, String> {

    @Query(value = "SELECT TOP 1 * FROM c ORDER BY c._ts DESC")
    List<BatteryData> findLatestReading();


}