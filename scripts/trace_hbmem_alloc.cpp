// Observe allocation requests without changing size, flags, or allocator.
#include <hbmem.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
struct Ledger { std::mutex lock; std::map<hbmem_addr_t,uint32_t> sizes; uint64_t live=0,peak=0; };
static Ledger &ledger() { static Ledger instance; return instance; }
extern "C" hbmem_addr_t hbmem_alloc(uint32_t size, uint64_t flag, const char *label) {
  using Function = hbmem_addr_t (*)(uint32_t, uint64_t, const char *);
  static auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "hbmem_alloc"));
  if (!original) { std::fprintf(stderr, "TRACE_HBMEM missing symbol\n"); std::abort(); }
  auto result = original(size, flag, label);
  auto &l=ledger();
  {
    std::lock_guard<std::mutex> guard(l.lock);
    if (result) { l.sizes[result]=size; l.live+=size; if(l.live>l.peak) l.peak=l.live; }
    std::fprintf(stderr, "TRACE_HBMEM bytes=%u flags=%llu label=%s success=%d live=%llu peak=%llu\n",
                 size, static_cast<unsigned long long>(flag), label ? label : "", result != 0,
                 static_cast<unsigned long long>(l.live),static_cast<unsigned long long>(l.peak));
  }
  if (!result) {
    for (const char *name : {"carveout","cma_reserved","ion_cma"}) {
      char path[160]; std::snprintf(path,sizeof(path),"/sys/kernel/debug/ion/heaps/%s",name);
      FILE *f=std::fopen(path,"r");
      if(f) { char line[1024]; std::fprintf(stderr,"TRACE_HEAP %s\n",name);
        while(std::fgets(line,sizeof(line),f)) std::fputs(line,stderr);
        std::fclose(f); }
    }
  }
  return result;
}
extern "C" void hbmem_free(hbmem_addr_t addr) {
  using Function=void (*)(hbmem_addr_t);
  static auto original=reinterpret_cast<Function>(dlsym(RTLD_NEXT,"hbmem_free"));
  if(!original) std::abort();
  original(addr);
  auto &l=ledger(); std::lock_guard<std::mutex> guard(l.lock);
  auto it=l.sizes.find(addr);
  if(it!=l.sizes.end()) { l.live-=it->second; l.sizes.erase(it); }
  std::fprintf(stderr,"TRACE_FREE live=%llu peak=%llu\n",static_cast<unsigned long long>(l.live),static_cast<unsigned long long>(l.peak));
}
