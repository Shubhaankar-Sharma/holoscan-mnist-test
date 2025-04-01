// profiler.hpp
#ifndef HOLOSCAN_PROFILER_HPP
#define HOLOSCAN_PROFILER_HPP

#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>  // NVTX3 header

// Define a custom domain for our profiler
struct holoscan_domain { static constexpr char const* name{"HOLOSCAN"}; };

class HoloscanProfiler {
public:
    // Singleton pattern
    static HoloscanProfiler& getInstance() {
        static HoloscanProfiler instance;
        return instance;
    }

    // CPU timing methods
    void startCpuTimer(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Create a named string for NVTX (avoid temporary strings)
        std::string event_name = name + " start";
        // Mark the start in NVTX
        nvtx3::mark_in<holoscan_domain>(event_name.c_str());
        
        cpu_timers_[name].push_back(std::chrono::high_resolution_clock::now());
    }

    void stopCpuTimer(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto stop_time = std::chrono::high_resolution_clock::now();
        auto& timer_stack = cpu_timers_[name];
        
        if (timer_stack.empty()) {
            std::cerr << "Error: Trying to stop timer '" << name << "' that wasn't started" << std::endl;
            return;
        }
        
        auto start_time = timer_stack.back();
        timer_stack.pop_back();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();
        cpu_durations_[name].push_back(duration);
        
        // Create a named string for NVTX (avoid temporary strings)
        std::string event_name = name + " end";
        // Mark the end in NVTX
        nvtx3::mark_in<holoscan_domain>(event_name.c_str());
    }

    // CUDA event timing methods
    void startCudaTimer(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Create a named string for NVTX (avoid temporary strings)
        std::string event_name = name + " start";
        
        // Create event attributes with the name and color
        nvtx3::event_attributes attr(event_name.c_str(), nvtx3::rgb{255, 0, 0});
        
        // Mark the start in NVTX
        nvtx3::mark_in<holoscan_domain>(attr);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        cuda_events_[name].push_back({start, stop});
    }

    void stopCudaTimer(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto& event_stack = cuda_events_[name];
        if (event_stack.empty()) {
            std::cerr << "Error: Trying to stop CUDA timer '" << name << "' that wasn't started" << std::endl;
            return;
        }
        
        auto& events = event_stack.back();
        cudaEventRecord(events.second);
        cudaEventSynchronize(events.second);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, events.first, events.second);
        cuda_durations_[name].push_back(milliseconds);
        
        event_stack.pop_back();
        
        // Create a named string for NVTX (avoid temporary strings)
        std::string event_name = name + " end";
        
        // Create event attributes with the name and color
        nvtx3::event_attributes attr(event_name.c_str(), nvtx3::rgb{255, 0, 0});
        
        // Mark the end in NVTX
        nvtx3::mark_in<holoscan_domain>(attr);
    }

    // Add a method for direct NVTX markers
    void markEvent(const std::string& name) {
        // Create a marker (instantaneous event) in our domain
        nvtx3::mark_in<holoscan_domain>(name.c_str());
    }

    // Report generation
    void printReport() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::cout << "\n===== HOLOSCAN PROFILER REPORT =====\n";
        
        // CPU timings
        std::cout << "\n----- CPU TIMINGS -----\n";
        for (const auto& [name, durations] : cpu_durations_) {
            if (durations.empty()) continue;
            
            double total = 0.0;
            double min = durations[0];
            double max = durations[0];
            
            for (const auto& duration : durations) {
                total += duration;
                min = std::min(min, static_cast<double>(duration));
                max = std::max(max, static_cast<double>(duration));
            }
            
            double avg = total / durations.size();
            
            std::cout << "Timer: " << name << "\n";
            std::cout << "  Count: " << durations.size() << "\n";
            std::cout << "  Avg: " << avg << " µs\n";
            std::cout << "  Min: " << min << " µs\n";
            std::cout << "  Max: " << max << " µs\n";
            std::cout << "  Total: " << total << " µs\n\n";
        }
        
        // CUDA timings
        std::cout << "\n----- CUDA TIMINGS -----\n";
        for (const auto& [name, durations] : cuda_durations_) {
            if (durations.empty()) continue;
            
            double total = 0.0;
            double min = durations[0];
            double max = durations[0];
            
            for (const auto& duration : durations) {
                total += duration;
                min = std::min(min, static_cast<double>(duration));
                max = std::max(max, static_cast<double>(duration));
            }
            
            double avg = total / durations.size();
            
            std::cout << "Timer: " << name << "\n";
            std::cout << "  Count: " << durations.size() << "\n";
            std::cout << "  Avg: " << avg << " ms\n";
            std::cout << "  Min: " << min << " ms\n";
            std::cout << "  Max: " << max << " ms\n";
            std::cout << "  Total: " << total << " ms\n\n";
        }
    }

    void saveReportToFile(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
            return;
        }
        
        file << "Timer,Type,Count,Avg,Min,Max,Total\n";
        
        // CPU timings
        for (const auto& [name, durations] : cpu_durations_) {
            if (durations.empty()) continue;
            
            double total = 0.0;
            double min = durations[0];
            double max = durations[0];
            
            for (const auto& duration : durations) {
                total += duration;
                min = std::min(min, static_cast<double>(duration));
                max = std::max(max, static_cast<double>(duration));
            }
            
            double avg = total / durations.size();
            
            file << name << ",CPU," << durations.size() << "," 
                 << avg << "," << min << "," << max << "," << total << "\n";
        }
        
        // CUDA timings
        for (const auto& [name, durations] : cuda_durations_) {
            if (durations.empty()) continue;
            
            double total = 0.0;
            double min = durations[0];
            double max = durations[0];
            
            for (const auto& duration : durations) {
                total += duration;
                min = std::min(min, static_cast<double>(duration));
                max = std::max(max, static_cast<double>(duration));
            }
            
            double avg = total / durations.size();
            
            file << name << ",CUDA," << durations.size() << "," 
                 << avg << "," << min << "," << max << "," << total << "\n";
        }
        
        file.close();
    }

    // Reset all timers
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        cpu_timers_.clear();
        cpu_durations_.clear();
        
        // Clean up CUDA events
        for (auto& [name, event_stack] : cuda_events_) {
            for (auto& [start, stop] : event_stack) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
        }
        cuda_events_.clear();
    }

    // Destructor to clean up CUDA events
    ~HoloscanProfiler() {
        for (auto& [name, event_stack] : cuda_events_) {
            for (auto& [start, stop] : event_stack) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
        }
    }

private:
    HoloscanProfiler() = default;
    HoloscanProfiler(const HoloscanProfiler&) = delete;
    HoloscanProfiler& operator=(const HoloscanProfiler&) = delete;

    std::mutex mutex_;
    std::map<std::string, std::vector<std::chrono::high_resolution_clock::time_point>> cpu_timers_;
    std::map<std::string, std::vector<double>> cpu_durations_;
    
    std::map<std::string, std::vector<std::pair<cudaEvent_t, cudaEvent_t>>> cuda_events_;
    std::map<std::string, std::vector<float>> cuda_durations_;
};

// Convenient macros for profiling
#define PROFILE_CPU_START(name) HoloscanProfiler::getInstance().startCpuTimer(name)
#define PROFILE_CPU_STOP(name) HoloscanProfiler::getInstance().stopCpuTimer(name)
#define PROFILE_CUDA_START(name) HoloscanProfiler::getInstance().startCudaTimer(name)
#define PROFILE_CUDA_STOP(name) HoloscanProfiler::getInstance().stopCudaTimer(name)
#define PROFILE_NVTX_MARK(name) HoloscanProfiler::getInstance().markEvent(name)

// NVTX-specific convenience macros
#define NVTX_RANGE(name) nvtx3::scoped_range_in<holoscan_domain> _nvtx_range{name}
#define NVTX_MARK(name) nvtx3::mark_in<holoscan_domain>(name)

// Scope-based RAII profiler
class ScopedCpuProfiler {
public:
    explicit ScopedCpuProfiler(const std::string& name) : name_(name) {
        HoloscanProfiler::getInstance().startCpuTimer(name_);
    }
    
    ~ScopedCpuProfiler() {
        HoloscanProfiler::getInstance().stopCpuTimer(name_);
    }
    
private:
    std::string name_;
};

class ScopedCudaProfiler {
public:
    explicit ScopedCudaProfiler(const std::string& name) : name_(name) {
        HoloscanProfiler::getInstance().startCudaTimer(name_);
    }
    
    ~ScopedCudaProfiler() {
        HoloscanProfiler::getInstance().stopCudaTimer(name_);
    }
    
private:
    std::string name_;
};

// Direct NVTX scoped range profiler 
class ScopedNvtxProfiler {
public:
    explicit ScopedNvtxProfiler(const std::string& name) 
        // Create a scoped range directly as a member variable (stack allocation)
        : range_{name.c_str()}
    {
    }
    
private:
    // This must be a direct member, not a pointer
    nvtx3::scoped_range_in<holoscan_domain> range_;
};

#define PROFILE_CPU_SCOPE(name) ScopedCpuProfiler scoped_cpu_profiler_##__LINE__(name)
#define PROFILE_CUDA_SCOPE(name) ScopedCudaProfiler scoped_cuda_profiler_##__LINE__(name)
#define PROFILE_NVTX_SCOPE(name) ScopedNvtxProfiler scoped_nvtx_profiler_##__LINE__(name)

// Use the NVTX function range macro to automatically create a range for a function
#define FUNC_RANGE() NVTX3_FUNC_RANGE_IN(holoscan_domain)

#endif // HOLOSCAN_PROFILER_HPP